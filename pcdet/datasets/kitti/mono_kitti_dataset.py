import copy
import pickle
import numpy as np
import torch
from skimage import io

from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti, depth_map_utils
from pcdet.datasets.mono_dataset_template import MonoDatasetTemplate
from .stereo_kitti_dataset import StereoKittiDataset

class MonoKittiDataset(MonoDatasetTemplate, StereoKittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

    def __getitem__(self, index):
        assert self.dataset_cfg.FOV_POINTS_ONLY
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        # if self.dataset_cfg.get('debug', False):
        #     index = 3717 % len(self.kitti_infos)

        assert not self.boxes_gt_in_cam2_view
        assert not self.cat_reflect

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        raw_points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)
        calib_ori = copy.deepcopy(calib)

        pts_rect = calib.lidar_to_rect(raw_points[:, 0:3])
        reflect = raw_points[:, 3:4]

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            pts_rect = pts_rect[fov_flag]
            reflect = reflect[fov_flag]

        # load images
        left_img = self.get_image(info['image']['image_idx'], 2)
        right_img = self.get_image(info['image']['image_idx'], 3)

        # convert camera-view points into pseudo lidar points
        # see code in calibration_kitti.py
        # right: [x] --> [-y]
        # up: [-y] --> [z]
        # front: [z] --> [x]
        if self.cat_reflect:
            input_points = np.concatenate([calib.rect_to_lidar_pseudo(pts_rect), reflect], 1)
        else:
            input_points = calib.rect_to_lidar_pseudo(pts_rect)
        input_dict = {
            'points': input_points,
            'frame_id': sample_idx,
            'calib': calib,
            'calib_ori': calib_ori,
            'left_img': left_img,
            'right_img': right_img,
            'image_shape': left_img.shape
        }

        if 'annos' in info:
            annos = info['annos']
            if self.use_van:
                # Car 14357, Van 1297
                annos['name'][annos['name'] == 'Van'] = 'Car'
            if self.use_person_sitting:
                # Ped 2207, Person_sitting 56
                annos['name'][annos['name'] == 'Person_sitting'] = 'Pedestrian'
            full_annos = annos
            ignored_annos = common_utils.collect_ignored_with_name(full_annos, name=['DontCare'])  # only bbox is useful
            annos = common_utils.drop_info_with_name(full_annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_annots_boxes_2d = annos['bbox']
            gt_boxes_2d_ignored = ignored_annos['bbox']
            gt_truncated = annos['truncated']
            gt_occluded = annos['occluded']
            gt_difficulty = annos['difficulty']
            gt_index = annos['index']
            image_shape = left_img.shape

            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                gt_boxes_camera, calib, pseudo_lidar=True, pseudo_cam2_view=self.boxes_gt_in_cam2_view)
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'gt_boxes_2d_ignored': gt_boxes_2d_ignored,
                'gt_annots_boxes_2d': gt_annots_boxes_2d,
                'gt_truncated': gt_truncated,
                'gt_occluded': gt_occluded,
                'gt_difficulty': gt_difficulty,
                'gt_index': gt_index,
                'image_idx': index
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict
