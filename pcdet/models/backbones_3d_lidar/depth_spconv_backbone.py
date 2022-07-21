import torch.nn as nn
import torch.nn.functional as F
from .spconv_backbone import VoxelBackBone4x
from ..backbones_3d_stereo.submodule import convbn_3d

class DepthVoxelBackBone4x(VoxelBackBone4x):
    def __init__(self, model_cfg, **kwargs):
        super(DepthVoxelBackBone4x, self).__init__(model_cfg, **kwargs)

        self.GN = self.model_cfg.get('GN', False)
        if self.model_cfg.get('VOXEL_PRED', False):
            self.pred_occupancy = self.build_voxel_pred_module()
        
        if self.model_cfg.get('DEPTH_PRED', False):
            self.downsample_disp = self.model_cfg.get('downsample_disp', 4)
            self.voxel_occupancy_downsample_disp = self.model_cfg.get('voxel_occupancy_downsample_disp', 2)
            self.pred_occupancy = self.build_depth_pred_module()
            crop_x1, crop_x2, crop_y1, crop_y2 = 0, 1248, 0, 320
            self.coordinates_psv = self.prepare_coordinates_psv(crop_x1, crop_x2, crop_y1, crop_y2, 
                img_height=320, img_width=1248, 
                downsample_disp=(self.downsample_disp, self.downsample_disp, self.voxel_occupancy_downsample_disp))

    def build_depth_pred_module(self):
        return nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 3, 1, 1, bias=True),
            nn.Upsample(size=(288, 320, 1248), mode='trilinear', align_corners=True))

    def build_voxel_pred_module(self, upsample_ratio=(2,2,2)):
        return nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 3, 1, 1, bias=True),
            nn.Upsample(scale_factor=upsample_ratio, mode='trilinear', align_corners=True))

    def prepare_coordinates_psv(self, crop_x1, crop_x2, crop_y1, crop_y2, img_height, img_width, downsample_disp):
        us = torch.linspace(crop_y1 + 0.5 * downsample_disp[0], crop_y2 - 0.5 * downsample_disp[0], img_height // downsample_disp[0], dtype=torch.float32, device='cuda') # height
        vs = torch.linspace(crop_x1 + 0.5 * downsample_disp[1], crop_x2 - 0.5 * downsample_disp[1], img_width // downsample_disp[1], dtype=torch.float32, device='cuda') # width 
        if downsample_disp[2] == 4:
            ds = self.downsampled_depth.cuda()
        elif downsample_disp[2] == 2:
            ds = self.downsampledx2_depth.cuda()
        elif downsample_disp[2] == 1:
            ds = self.depth.cuda()
        ds, us, vs = torch.meshgrid(ds, us, vs)
        coordinates_psv = torch.stack([vs, us, ds], dim=-1)
        return coordinates_psv

    def forward(self, batch_dict):
        batch_dict = super(DepthVoxelBackBone4x, self).forward(batch_dict)

        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        batch_dict['volume_features'] = spatial_features

        spatial_features = F.interpolate(spatial_features, scale_factor=(4, 1, 1), mode='trilinear', align_corners=True)

        if self.model_cfg.get('VOXEL_PRED', False):
            batch_dict['voxel_occupancy'] = self.pred_occupancy(spatial_features)

        if self.model_cfg.get('DEPTH_PRED', False):
            from IPython import embed; embed()

        return batch_dict
