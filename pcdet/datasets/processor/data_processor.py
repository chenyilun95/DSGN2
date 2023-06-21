# From OpenPCDet. https://github.com/open-mmlab/OpenPCDet
# point cloud processor for masking, shuffling, and other transformations

from functools import partial
import numpy as np

from pcdet.utils import box_utils, common_utils
import torch

class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if 'completion_points' in data_dict:
            mask = common_utils.mask_points_by_range(data_dict['completion_points'], self.point_cloud_range)
            data_dict['completion_points'] = data_dict['completion_points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            for key in ['gt_names', 'gt_truncated', 'gt_occluded', 'gt_difficulty', 'gt_index', 'gt_boxes_no3daug']:
                if key in data_dict:
                    data_dict[key] = data_dict[key][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_projected_voxels(self, data_dict=None, config=None):
        """
        Ensure the voxel in forward coordinates for learning depth. None 3d augs should be applied
        """

        if data_dict is None:
            return partial(self.transform_points_to_projected_voxels, config=config)

        points = data_dict.get('completion_points', data_dict.get('points_no3daug', data_dict['points']) )
        calib = data_dict['calib']

        from pcdet.utils import common_utils, box_utils, depth_map_utils
        from pcdet.utils.calibration_kitti import Calibration
        
        self.X_MIN, self.Y_MIN, self.Z_MIN = self.point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = self.point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = config.VOXEL_SIZE
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
        grid_size = np.round(grid_size).astype(np.int64)
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = grid_size.tolist()

        sample_rate = (1, 1, 1)
        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        # point cloud -> voxelized point cloud
        voxel_sizes = np.asarray([ self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE ])[None]
        image_shape = (data_dict['left_img'].shape[:2] if 'left_img' in data_dict else data_dict['left_imgs'].shape[1:3])
        voxelized_points = ((points[:, :3] - np.asarray([self.X_MIN, self.Y_MIN, self.Z_MIN])) / voxel_sizes).astype(int) * voxel_sizes + \
            voxel_sizes / 2. + np.asarray([self.X_MIN, self.Y_MIN, self.Z_MIN]) # ensure voxelized to grid centers
        rect_voxelized_points = Calibration.lidar_pseudo_to_rect(voxelized_points[:, :3])
        voxelized_depth_gt_img, voxelized_deviation_gt_img = depth_map_utils.points_to_depth_deviation_map(rect_voxelized_points, image_shape, data_dict['calib'])

        # voxel grid
        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        dim1, dim2, dim3, _ = coordinates_3d.shape
        coordinates_3d = coordinates_3d.reshape(-1, 3)

        coordinates_3d = coordinates_3d.numpy()

        rect_points = Calibration.lidar_pseudo_to_rect(coordinates_3d)
        pts_img, pts_depth = calib.rect_to_img(rect_points[:, :3])
        iy, ix = np.round(pts_img[:, 1]).astype(np.int64), np.round(pts_img[:, 0]).astype(np.int64)
        
        voxels_in_ray = (iy >= 0) & (ix >= 0) & (iy < voxelized_depth_gt_img.shape[0]) & (ix < voxelized_depth_gt_img.shape[1])
        voxels_in_ray = np.where(voxels_in_ray)[0]

        valid_voxel = voxelized_depth_gt_img[iy[voxels_in_ray], ix[voxels_in_ray]] > 0.00001
        voxels_in_ray = voxels_in_ray[valid_voxel]

        valid_occupancy = (voxelized_depth_gt_img[iy[voxels_in_ray], ix[voxels_in_ray]] >= pts_depth[voxels_in_ray] - self.VOXEL_X_SIZE / 2.) \
            & (voxelized_depth_gt_img[iy[voxels_in_ray], ix[voxels_in_ray]] <= pts_depth[voxels_in_ray] + self.VOXEL_X_SIZE / 2.)
        
        if config.get('POS_WEIGHT', False):
            data_dict['norm_dist'] = np.linalg.norm(rect_points[voxels_in_ray], axis=1)

        data_dict['voxels_in_ray'] = voxels_in_ray
        data_dict['occupany_of_voxels_in_ray'] = valid_occupancy.astype(np.float32)

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        data_dict['voxel_size'] = self.voxel_size
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
