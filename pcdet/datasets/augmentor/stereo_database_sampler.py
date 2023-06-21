# Modified from OpenPCDet. https://github.com/open-mmlab/OpenPCDet
# DataBaseSampler is used for cut-and-paste augmentation for LiDAR Point clouds.

import pickle

from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from skimage import io
import cv2
import torch
import math

import numpy as np

def warp(img, bbox, target_bbox):
    bbox = bbox.flatten()
    center = tuple(bbox.reshape(2, 2).mean(axis=0))
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    angle = 0.
    rect = center, (w, h), angle
    box = cv2.boxPoints(rect)

    target_bbox = target_bbox.flatten()
    center = tuple(target_bbox.reshape(2, 2).mean(axis=0))
    w, h = target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1]
    angle = 0.
    rect = center, (w, h), angle
    target_box = cv2.boxPoints(rect)

    m = cv2.getPerspectiveTransform(box, target_box)
    warped = cv2.warpPerspective(img, m, (np.ceil(target_bbox[2]).astype(int), np.ceil(target_bbox[3]).astype(int)), flags=cv2.INTER_LINEAR)
    return warped

def query_bbox_overlaps(boxes, query_boxes):
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy() # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * \
            (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
            (query_boxes[:, 3] - query_boxes[:, 1] + 1)

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t()) + 1).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t()) + 1).clamp(min=0)
    # ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    # return how much overlap of the query boxes instead of full mask
    overlaps = iw * ih / query_areas.view(1, -1)
    return out_fn(overlaps)

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        self.filter_occlusion_overlap = sampler_cfg.get('filter_occlusion_overlap', 1.)
        self.far_to_near = getattr(self.sampler_cfg, 'far_to_near', False)
        self.stop_epoch = getattr(self.sampler_cfg, 'stop_epoch', None)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def set_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']

        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices

        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_pseudo_to_rect(gt_boxes[:, 0:3])
        # height at the point [x, 0, z], direction is upward [a, b, c]~=[0, -1, 1]
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        # set the height of the box to be zero plane
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar_pseudo(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height
    
    @staticmethod
    def put_boxes_on_pseudo_road_planes(gt_boxes, points, calib, pre_gt_boxes):
        from scipy import stats
        import scipy.interpolate as interpolate

        # if len(pre_gt_boxes) < 3:
        #     use_linear_regression = True
        #     res = stats.linregress(pre_gt_boxes[:, 0], pre_gt_boxes[:, 2])
        # else:
        #     use_linear_regression = False
        use_linear_regression = False

        # use already existed boxes as interpolation of their avg height from ground planes
        if len(pre_gt_boxes) < 2:
            use_interpolate_ground_plane = False
        else:
            pre_gt_boxes = pre_gt_boxes.copy()
            sort_idxs = pre_gt_boxes[:, 0].argsort()
            pre_gt_boxes = pre_gt_boxes[sort_idxs]

            # # avoid closest boxes 
            # too_close_boxes = pre_gt_boxes[1:, 0] - pre_gt_boxes[:-1, 0] > 1.5 
            # too_close_boxes = np.concatenate([[True], too_close_boxes], axis=0)
            # pre_gt_boxes = pre_gt_boxes[too_close_boxes]

            if len(pre_gt_boxes) < 2:
                use_interpolate_ground_plane = False
            else:
                linear_planes = interpolate.interp1d(pre_gt_boxes[:, 0], pre_gt_boxes[:, 2]) #, bounds_error=False, fill_value="extrapolate")
                use_interpolate_ground_plane = True

        # if not use_interpolate_ground_plane: # otherwise use the minimum points
        bev_boxes = gt_boxes.copy() # (x, y, z, l, w, h, theta) in lidar coords, (x, y, z, l, h, w, theta) in rect coord
        bev_boxes[:, 5] = 1000. # height -> max
        bev_boxes[:, 2] = -100.
        # bev_boxes[:, [3, 4]] = bev_boxes[:, [3, 4]].clip(min=2.)
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(points[:, 0:3]), torch.from_numpy(bev_boxes[:, :7])
        ).numpy()

        mv_height = np.zeros((len(gt_boxes)))
        for i in range(len(gt_boxes)):
            if use_interpolate_ground_plane and linear_planes.x.min() < gt_boxes[i, 0] < linear_planes.x.max(): # interpolation
                mv_height[i] = gt_boxes[i, 2] - linear_planes(gt_boxes[i, 0])
            else:
                if point_indices[i].sum() > 0:
                    bev_min_height = points[point_indices[i] > 0].min(0)[2]
                else:
                    # find nearest points
                    nearest_point = points[np.linalg.norm(points[:,:2] - gt_boxes[i:i+1, :2], axis=-1).argmin()]
                    point_indice = np.linalg.norm(points[:, :2] - nearest_point[None, :2], axis=-1) < 2. # 2 meters minimum
                    bev_min_height = points[point_indice].min(0)[2]
                mv_height[i] = gt_boxes[i, 2] - bev_min_height

            # if use_linear_regression: # fix large gap with linear regression
            #     regress_z = gt_boxes[i, 2] - (res.intercept + res.slope * gt_boxes[i, 0])
            #     if mv_height[i] - regress_z > 2.:
            #         mv_height[i] = regress_z

        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        # sorted by distance
        if getattr(self, 'far_to_near', False):
            sorted_inds = (-sampled_gt_boxes[:,0]).argsort() # near to far !
        else:
            sorted_inds = (sampled_gt_boxes[:,0]).argsort() # near to far !
        sampled_gt_boxes = sampled_gt_boxes[sorted_inds]
        total_valid_sampled_dict = [total_valid_sampled_dict[idx] for idx in sorted_inds]

        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']

        gt_difficulty = data_dict['gt_difficulty'][gt_boxes_mask]
        gt_occluded = data_dict['gt_occluded'][gt_boxes_mask]
        gt_index = data_dict['gt_index'][gt_boxes_mask]
        gt_truncated = data_dict['gt_truncated'][gt_boxes_mask]

        # bbox_2d need to store the shift between original calib and shift
        # left_bbox_2d = np.asarray([info['bbox'] for info in total_valid_sampled_dict])
        # ori_sampled_gt_boxes_rect = box_utils.boxes3d_lidar_to_kitti_camera(sampled_gt_boxes, pseudo_lidar=True)
        # ori_left_bbox_img = box_utils.boxes3d_kitti_camera_to_imageboxes(ori_sampled_gt_boxes_rect, calib=data_dict['calib'])

        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            if 'road_plane' in data_dict:
                sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                    sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
                )
                # data_dict.pop('calib')
                data_dict.pop('road_plane')
            else:
                sampled_gt_boxes, mv_height = self.put_boxes_on_pseudo_road_planes(
                    sampled_gt_boxes, points, data_dict['calib'], data_dict['gt_boxes']
                )
                raise ValueError('Please use the pre-computed road planes for better results.')

        left_img = data_dict['left_img']
        right_img = data_dict['right_img']

        calib = data_dict['calib']
        sampled_gt_box_corners = box_utils.boxes_to_corners_3d(sampled_gt_boxes) # already move down to road
        N, _, _ = sampled_gt_box_corners.shape
        sampled_gt_box_corners_rect = calib.lidar_pseudo_to_rect(sampled_gt_box_corners.reshape(-1, 3))
        left_pts_img, left_pts_depth = calib.rect_to_img(sampled_gt_box_corners_rect) # left
        right_pts_img, right_pts_depth = calib.rect_to_img(sampled_gt_box_corners_rect, right=True) # left

        left_pts_img = left_pts_img.reshape(N, 8, 2)
        right_pts_img = right_pts_img.reshape(N, 8, 2)

        left_bbox_img = np.concatenate([left_pts_img.min(axis=1), left_pts_img.max(axis=1)], axis=1) # slightly larger bbox
        right_bbox_img = np.concatenate([right_pts_img.min(axis=1), right_pts_img.max(axis=1)], axis=1)

        # # move shifts in 2D
        # shift_2d = (left_bbox_img - ori_left_bbox_img)
        # left_bbox_2d += shift_2d

        left_bbox_img_int = left_bbox_img.astype(int)
        right_bbox_img_int = right_bbox_img.astype(int)

        left_bbox_img_int[:, [0, 2]] = left_bbox_img_int[:, [0, 2]].clip(min=0, max=left_img.shape[1] - 1)
        left_bbox_img_int[:, [1, 3]] = left_bbox_img_int[:, [1, 3]].clip(min=0, max=left_img.shape[0] - 1)
        right_bbox_img_int[:, [0, 2]] = right_bbox_img_int[:, [0, 2]].clip(min=0, max=right_img.shape[1] - 1)
        right_bbox_img_int[:, [1, 3]] = right_bbox_img_int[:, [1, 3]].clip(min=0, max=right_img.shape[0] - 1)

        left_cropped_bbox = left_bbox_img - left_bbox_img_int[:, [0, 1, 0, 1]]
        right_cropped_bbox = right_bbox_img - right_bbox_img_int[:, [0, 1, 0, 1]]

        points_img, points_depth = calib.rect_to_img( calib.lidar_pseudo_to_rect(points) )
        if self.sampler_cfg.get('remove_overlapped', True):
            non_overlapped_mask = np.ones(len(points_img), dtype=bool)

        obj_points_list = []
        obj_points_img_list = []
        sampled_gt_indices = []
        sampled_gt_difficulty = []
        sampled_mask = np.zeros((0,), dtype=bool)
        check_overlap_boxes = np.zeros((0, 4), dtype=float)
        check_overlap_boxes = np.append(check_overlap_boxes, data_dict['gt_annots_boxes_2d'], axis=0)
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            
            ### read points
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])
            obj_points[:, :3] += info['box3d_lidar'][:3]
            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            ### read images
            cropped_left_img = io.imread(self.root_path / info['cropped_left_img_path'])
            cropped_right_img = io.imread(self.root_path / info['cropped_right_img_path'])
            cropped_left_bbox = info['cropped_left_bbox']
            cropped_right_bbox = info['cropped_right_bbox']
            cropped_left_bbox[[2,3]] -= 1 # fix bug of prepare dataset
            cropped_right_bbox[[2,3]] -= 1

            # if cropped_left_bbox[0] < 0. or cropped_left_bbox[1] < 0. or cropped_left_bbox[2] >= cropped_left_img.shape[1] - 1 or cropped_left_bbox[3] >= cropped_left_img.shape[0] - 1:
            #     continue
            left_warped_img = warp(cropped_left_img, cropped_left_bbox, left_cropped_bbox[idx])
            max_bbox_h = min(left_warped_img.shape[0], left_img.shape[0]-left_bbox_img_int[idx, 1])
            max_bbox_w = min(left_warped_img.shape[1], left_img.shape[1]-left_bbox_img_int[idx, 0])

            sampled_crop_box = np.asarray([[left_bbox_img_int[idx, 0], left_bbox_img_int[idx, 1], left_bbox_img_int[idx, 0]+max_bbox_w, left_bbox_img_int[idx, 1]+max_bbox_h]], dtype=float)
            if self.filter_occlusion_overlap < 1.:
                overlap_with_fg = query_bbox_overlaps(sampled_crop_box, check_overlap_boxes)
                # print(overlap_with_fg)
                if np.prod(overlap_with_fg.shape) > 0 and min(overlap_with_fg.max(), 1.) > self.filter_occlusion_overlap:
                    sampled_mask = np.append(sampled_mask, False)
                    continue
            
            sampled_mask = np.append(sampled_mask, True)
            check_overlap_boxes = np.append(check_overlap_boxes, sampled_crop_box, axis=0)
            
            obj_points_list.append(obj_points)
            obj_points_img_list.append( calib.rect_to_img( calib.lidar_pseudo_to_rect(obj_points[:, :3]) )[0] )
            sampled_gt_indices.append(info['gt_idx'])
            sampled_gt_difficulty.append(info['difficulty'])

            left_img[ left_bbox_img_int[idx, 1]:left_bbox_img_int[idx, 1]+max_bbox_h, left_bbox_img_int[idx, 0]:left_bbox_img_int[idx, 0]+max_bbox_w ] = left_warped_img[:max_bbox_h, :max_bbox_w]
            # print(f'left: origin size {left_warped_img.shape[1]}x{left_warped_img.shape[0]} final cropped box size{max_bbox_w}x{max_bbox_h}')

            if self.sampler_cfg.get('remove_overlapped', True):
                non_overlapped_mask &= ( (points_img[:, 0] <= left_bbox_img_int[idx, 0]) | (points_img[:, 0] >= left_bbox_img_int[idx, 2]) | \
                    (points_img[:, 1] < left_bbox_img_int[idx, 1]) | (points_img[:, 1] >= left_bbox_img_int[idx, 3]) )
                for j in range(len(obj_points_list) - 1): # exclude itself
                    non_overlapped_obj_mask = ( (obj_points_img_list[j][:, 0] <= left_bbox_img_int[idx, 0]) | (obj_points_img_list[j][:, 0] >= left_bbox_img_int[idx, 2]) | \
                        (obj_points_img_list[j][:, 1] < left_bbox_img_int[idx, 1]) | (obj_points_img_list[j][:, 1] >= left_bbox_img_int[idx, 3]) )
                    obj_points_img_list[j] = obj_points_img_list[j][non_overlapped_obj_mask]
                    obj_points_list[j] = obj_points_list[j][non_overlapped_obj_mask]

            right_warped_img = warp(cropped_right_img, cropped_right_bbox, right_cropped_bbox[idx])
            max_bbox_h = min(right_warped_img.shape[0], right_img.shape[0]-right_bbox_img_int[idx, 1])
            max_bbox_w = min(right_warped_img.shape[1], right_img.shape[1]-right_bbox_img_int[idx, 0])
            right_img[ right_bbox_img_int[idx, 1]:right_bbox_img_int[idx, 1]+max_bbox_h, right_bbox_img_int[idx, 0]:right_bbox_img_int[idx, 0]+max_bbox_w ] = right_warped_img[:max_bbox_h, :max_bbox_w]
            # print(f'right: origin size {right_warped_img.shape[1]}x{right_warped_img.shape[0]} final cropped box size{max_bbox_w}x{max_bbox_h}')

        if self.sampler_cfg.get('remove_overlapped', True):
            points = points[non_overlapped_mask]

        sampled_gt_boxes = sampled_gt_boxes[sampled_mask]
        total_valid_sampled_dict = [total_valid_sampled_dict[idx] for idx in np.nonzero(sampled_mask)[0]]

        if len(obj_points_list) == 0:
            return data_dict

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points[:, :3], points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        data_dict['left_img'] = left_img
        data_dict['right_img'] = right_img

        data_dict['gt_boxes_mask'] = np.ones(len(gt_boxes), dtype=bool)
        data_dict['gt_difficulty'] = np.concatenate((gt_difficulty, np.array(sampled_gt_difficulty)))
        data_dict['gt_occluded'] = np.concatenate((gt_occluded, np.zeros(len(gt_boxes) - len(gt_occluded))))
        data_dict['gt_truncated'] = np.concatenate((gt_truncated, np.zeros(len(gt_boxes) - len(gt_truncated))))
        data_dict['gt_index'] = np.concatenate((gt_index, np.array(sampled_gt_indices)))
        # data_dict['gt_annots_boxes_2d'] = np.concatenate([data_dict['gt_annots_boxes_2d'], left_bbox_2d[sampled_mask]]) # TODO(hack) projected 3d boxes is not 2d bbox
        data_dict['gt_boxes_2d_ignored'] = np.concatenate([data_dict['gt_boxes_2d_ignored'], left_bbox_img[sampled_mask]]) # TODO(hack) projected 3d boxes is not 2d bbox

        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        if data_dict['calib'].flipped:
            print('flipped, skip gt_sampling')
            return data_dict
        
        if np.random.rand() > self.sampler_cfg.get('ratio', 0.6):
            return data_dict
        
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            print("Remove ground-truth data sampling at last 5 epochs.")
            return data_dict
        
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2

                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        return data_dict
