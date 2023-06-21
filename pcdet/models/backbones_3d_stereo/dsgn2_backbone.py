# DSGN++ backbone (fv+tv)
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from mmdet.models.builder import build_backbone, build_neck
from . import submodule
from .submodule import convbn_3d, convbn, feature_extraction_neck
from .cost_volume import BuildCostVolume
from pcdet.ops.build_geometry_volume import build_geometry_volume
from pcdet.ops.build_dps_geometry_volume import build_dps_geometry_volume
from pcdet.utils.torch_utils import *

class DSGN2Backbone(nn.Module):
    def __init__(self, model_cfg, class_names, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # general config
        self.class_names = class_names
        self.GN = model_cfg.GN
        self.fullres_stereo_feature = model_cfg.feature_neck.with_upconv

        # stereo config
        self.mono = getattr(model_cfg, 'mono', False)
        self.maxdisp = model_cfg.maxdisp
        self.downsample_disp = model_cfg.downsample_disp
        self.voxel_occupancy_downsample_disp = getattr(model_cfg, 'voxel_occupancy_downsample_disp', self.downsample_disp)
        self.downsampled_depth_offset = model_cfg.downsampled_depth_offset
        self.num_hg = getattr(model_cfg, 'num_hg', 1)
        self.use_stereo_out_type = getattr(model_cfg, 'use_stereo_out_type', 'feature')
        self.sup_geometry = getattr(model_cfg, 'sup_geometry', 'volume')
        assert self.sup_geometry in ['volume', 'pooledvolume']
        assert self.use_stereo_out_type in ["feature", "cost", "prob"]

        # volume construction config
        self.cat_img_feature = model_cfg.cat_img_feature
        self.cat_right_img_feature = getattr(model_cfg, 'cat_right_img_feature', False)
        self.rpn3d_dim = model_cfg.rpn3d_dim
        self.voxel_occupancy = getattr(model_cfg, 'voxel_occupancy', False)
        self.voxel_pred_convs = getattr(model_cfg, 'voxel_pred_convs', 1)
        self.voxel_pred_hgs = getattr(model_cfg, 'voxel_pred_hgs', 0)
        self.voxel_occupancy_upsample = getattr(model_cfg, 'voxel_occupancy_upsample', (2,2,2))
        self.drop_psv = getattr(model_cfg, 'drop_psv', False)
        self.geometry_volume_shift = getattr(model_cfg, 'geometry_volume_shift', 1)

        self.inv_smooth_psv = getattr(model_cfg, 'inv_smooth_psv', -1)
        self.inv_smooth_geo = getattr(model_cfg, 'inv_smooth_geo', -1)
        self.drop_psv_loss = getattr(model_cfg, 'drop_psv_loss', False)
        self.squeeze_geo = getattr(model_cfg, 'squeeze_geo', False)

        # volume config
        self.num_3dconvs = model_cfg.num_3dconvs
        self.num_3dconvs_hg = getattr(model_cfg, 'num_3dconvs_hg', 0)
        self.cv_dim = model_cfg.cv_dim

        # feature extraction
        self.feature_backbone = build_backbone(model_cfg.feature_backbone)
    
        feature_backbone_pretrained = getattr(model_cfg, 'feature_backbone_pretrained', None)
        if feature_backbone_pretrained:
            self.feature_backbone.init_weights(pretrained=feature_backbone_pretrained)

        self.feature_neck = feature_extraction_neck(model_cfg.feature_neck)
        if getattr(model_cfg, 'sem_neck', None):
            self.sem_neck = build_neck(model_cfg.sem_neck)
        else:
            self.sem_neck = None

        if not self.drop_psv:
            # cost volume
            self.build_cost = BuildCostVolume(model_cfg.cost_volume)

            # stereo network
            CV_INPUT_DIM = self.build_cost.get_dim(self.feature_neck.stereo_dim[-1]) if not self.mono else self.cv_dim
            self.dres0 = nn.Sequential(
                convbn_3d(CV_INPUT_DIM, self.cv_dim, 1, 1, 0, gn=self.GN),
                nn.ReLU(inplace=True))
            self.dres1 = nn.Sequential(
                convbn_3d(self.cv_dim, self.cv_dim, 3, 1, 1, gn=self.GN))
            self.hg_stereo = nn.ModuleList()
            for _ in range(self.num_hg):
                self.hg_stereo.append(submodule.hourglass(self.cv_dim, gn=self.GN))

        self.front_surface_depth = self.model_cfg.get('front_surface_depth', False)
        if (not self.drop_psv and not self.drop_psv_loss) or self.front_surface_depth:
            # stereo predictions
            self.pred_stereo = nn.ModuleList()
            for _ in range(max(self.num_hg, 1)):
                self.pred_stereo.append(self.build_depth_pred_module(self.cv_dim if not self.front_surface_depth else self.rpn3d_dim))
            self.dispregression = submodule.disparityregression()

        if self.voxel_occupancy:
            self.pred_occupancy = self.build_voxel_pred_module(upsample_ratio=self.voxel_occupancy_upsample, voxel_pred_convs=self.voxel_pred_convs, voxel_pred_hgs=self.voxel_pred_hgs)

        # rpn3d convs
        if self.drop_psv:
            RPN3D_INPUT_DIM = 0
        else:
            RPN3D_INPUT_DIM = self.cv_dim if not (self.use_stereo_out_type != "feature") else 1
        
        if self.squeeze_geo:
            assert self.cat_img_feature
            RPN3D_INPUT_DIM += self.rpn3d_dim
            self.squeeze_geo_conv = nn.Sequential(
                convbn_3d(self.cv_dim * (2 if self.cat_right_img_feature else 1), self.cv_dim, 1, 1, 0, gn=self.GN and self.cv_dim >= 32),
                nn.ReLU(inplace=True),
                convbn_3d(self.cv_dim, self.rpn3d_dim, 3, 1, 1, gn=self.GN),
                nn.ReLU(inplace=True),
            )
        else:
            if self.cat_img_feature:
                RPN3D_INPUT_DIM += self.rpn3d_dim #self.feature_neck.sem_dim[-1]
            if self.cat_right_img_feature:
                RPN3D_INPUT_DIM += self.rpn3d_dim #self.feature_neck.sem_dim[-1]
            
        rpn3d_convs = []
        for i in range(self.num_3dconvs):
            rpn3d_convs.append(
                nn.Sequential(
                    convbn_3d(RPN3D_INPUT_DIM if i == 0 else self.rpn3d_dim,
                              self.rpn3d_dim, 1 if i == 0 else 3, 1, 0 if i == 0 else 1, gn=self.GN and self.rpn3d_dim >= 32),
                    nn.ReLU(inplace=True)))
        self.rpn3d_convs = nn.Sequential(*rpn3d_convs)
        
        if self.num_3dconvs_hg > 0:
            self.rpn3d_hgs = nn.ModuleList()
            for i in range(self.num_3dconvs_hg):
                self.rpn3d_hgs.append(submodule.hourglass(self.rpn3d_dim, gn=self.GN and self.rpn3d_dim >= 32, planes_mul=[2,2]))
        self.rpn3d_pool = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))

        # prepare tensors
        self.point_cloud_range = kwargs.get('stereo_point_cloud_range', point_cloud_range)
        self.voxel_size, self.grid_size = voxel_size, grid_size
        self.prepare_depth(self.point_cloud_range, in_camera_view=False)
        self.prepare_coordinates_3d(self.point_cloud_range, voxel_size, grid_size)
        self.max_crop_shape = kwargs.get('max_crop_shape', (320, 1248))
        if self.front_surface_depth:
            crop_x1, crop_x2, crop_y1, crop_y2 = 0, self.max_crop_shape[1], 0, self.max_crop_shape[0]
            self.coordinates_psv = self.prepare_coordinates_psv(crop_x1, crop_x2, crop_y1, crop_y2, 
                img_height=self.max_crop_shape[0], img_width=self.max_crop_shape[1], 
                downsample_disp=(self.downsample_disp, self.downsample_disp, self.voxel_occupancy_downsample_disp))

        self.init_params()

    def build_depth_pred_module(self, cv_dim):
        return nn.Sequential(
            convbn_3d(cv_dim, cv_dim, 3, 1, 1, gn=self.GN and cv_dim>=32),
            nn.ReLU(inplace=True),
            nn.Conv3d(cv_dim, 1, 3, 1, 1, bias=True),
            nn.Upsample(scale_factor=self.downsample_disp, mode='trilinear', align_corners=True))

    def build_voxel_pred_module(self, upsample_ratio=(2,2,2), voxel_pred_convs=1, voxel_pred_hgs=0):
        voxel_module = []
        for i in range(voxel_pred_convs):
            voxel_module.extend([
                convbn_3d(self.rpn3d_dim, self.rpn3d_dim, 3, 1, 1, gn=self.GN),
                nn.ReLU(inplace=True),
            ])
        for i in range(voxel_pred_hgs):
            voxel_module.extend([
                submodule.hourglass(self.rpn3d_dim, gn=self.GN),
            ])
        if upsample_ratio and upsample_ratio[0] > 1:
            voxel_module.append(nn.Upsample(scale_factor=tuple(upsample_ratio), mode='trilinear'))
        else:
            voxel_module.append(nn.Identity())
        return nn.Sequential(*voxel_module)

    def prepare_depth(self, point_cloud_range, in_camera_view=True):
        if in_camera_view:
            self.CV_DEPTH_MIN = point_cloud_range[2]
            self.CV_DEPTH_MAX = point_cloud_range[5]
        else:
            self.CV_DEPTH_MIN = point_cloud_range[0]
            self.CV_DEPTH_MAX = point_cloud_range[3]
        assert self.CV_DEPTH_MIN >= 0 and self.CV_DEPTH_MAX > self.CV_DEPTH_MIN
        depth_interval = (self.CV_DEPTH_MAX - self.CV_DEPTH_MIN) / self.maxdisp
        print('stereo volume depth range: {} -> {}, interval {}'.format(self.CV_DEPTH_MIN,
                                                                        self.CV_DEPTH_MAX, depth_interval))
        # prepare downsampled depth
        self.downsampled_depth = torch.zeros(
            (self.maxdisp // self.downsample_disp), dtype=torch.float32)
        for i in range(self.maxdisp // self.downsample_disp):
            self.downsampled_depth[i] = (
                i + self.downsampled_depth_offset) * self.downsample_disp * depth_interval + self.CV_DEPTH_MIN
        self.downsampledx2_depth = torch.zeros(
            (self.maxdisp // 2), dtype=torch.float32)
        for i in range(self.maxdisp // 2):
            self.downsampledx2_depth[i] = (
                i + self.downsampled_depth_offset) * 2 * depth_interval + self.CV_DEPTH_MIN
        # prepare depth
        self.depth = torch.zeros((self.maxdisp), dtype=torch.float32)
        for i in range(self.maxdisp):
            self.depth[i] = (
                i + 0.5) * depth_interval + self.CV_DEPTH_MIN

    def prepare_coordinates_3d(self, point_cloud_range, voxel_size, grid_size, sample_rate=(1, 1, 1)):
        self.X_MIN, self.Y_MIN, self.Z_MIN = point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = voxel_size
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = grid_size.tolist()

        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()

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

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.voxel_occupancy:
            torch.nn.init.normal_(self.pred_occupancy[0][0].weight, mean=0, std=0.01)
            # torch.nn.init.constant_(self.pred_occupancy[0][0].bias, -2.19)

    def pred_depth(self, depth_conv_module, cost1, img_shape):
        cost1 = depth_conv_module(cost1)
        if cost1.shape[2] != self.maxdisp:
            cost1 = F.interpolate(
                cost1, [self.maxdisp, *img_shape],
                mode='trilinear',
                align_corners=True)
        cost1 = torch.squeeze(cost1, 1)
        cost1_softmax = F.softmax(cost1, dim=1)
        pred1 = self.dispregression(cost1_softmax,
                                    depth=self.depth.cuda())
        return cost1, cost1_softmax, pred1

    def pred_voxel(self, voxel_conv_module, voxel):
        voxel_occupancy = voxel_conv_module(voxel)
        return voxel_occupancy

    def get_local_depth(self, d_prob):
        with torch.no_grad():
            mean_d = []
            for i in range(len(d_prob)):
                d = self.depth.cuda()[None, :, None, None]
                d_mul_p = d * d_prob[i:i+1]
                local_window = 5
                p_local_sum = 0
                for off in range(0, local_window):
                    cur_p = d_prob[i:i+1, off:off + d_prob.shape[1] - local_window + 1]
                    p_local_sum += cur_p
                max_indices = p_local_sum.max(1, keepdim=True).indices
                pd_local_sum_for_max = 0
                for off in range(0, local_window):
                    cur_pd = torch.gather(d_mul_p, 1, max_indices + off).squeeze(1)  # d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
                    pd_local_sum_for_max += cur_pd
                mean_d.append(pd_local_sum_for_max / torch.gather(p_local_sum, 1, max_indices).squeeze(1))
            mean_d = torch.cat(mean_d, dim=0)
        return mean_d

    def forward_2d(self, img):
        features = self.feature_backbone(img)
        features = [img] + list(features)
        return self.feature_neck(features)

    @staticmethod
    def compute_mapping(c3d, image_shape, calib_proj, depth_range, pose_transform=None):
        coord_img = project_rect_to_image(
            c3d,
            calib_proj,
            pose_transform)
        coord_img = torch.cat(
            [coord_img, c3d[..., 2:]], dim=-1)
        
        # TODO: crop augmentation
        crop_x1, crop_x2 = 0, image_shape[1]
        crop_y1, crop_y2 = 0, image_shape[0]
        # assert (crop_x1, crop_x2, crop_y1, crop_y2) == (0, 1248, 0, 320)
        norm_coord_img = (coord_img - torch.as_tensor([crop_x1, crop_y1, depth_range[0]], device=coord_img.device)) / torch.as_tensor(
            [crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1, depth_range[1] - depth_range[0]], device=coord_img.device)
        # resize to [-1, 1]
        norm_coord_img = norm_coord_img * 2. - 1.
        return coord_img, norm_coord_img

    def compute_disp_channels(self, voxel_disps, img_channels, inv_ratio=0.1):
        shift_channels = (img_channels - self.cv_dim + 1)
        voxel_disps_channels = F.interpolate(voxel_disps[None,None], (shift_channels,), mode='linear')[0,0]
        if inv_ratio > 0.:
            voxel_disps_channels = voxel_disps_channels ** inv_ratio
        voxel_disps_channels = voxel_disps_channels / ((voxel_disps_channels.max() - voxel_disps_channels.min()) / shift_channels)
        voxel_disps_channels -= voxel_disps_channels.min()
        voxel_disps_channels = shift_channels-1 - voxel_disps_channels.to(int).clamp(0, shift_channels-1)
        return voxel_disps_channels
    
    def build_3d_geometry_volume(self, RPN_feature, norm_coord_imgs, voxel_disps):
        if RPN_feature.shape[1] <= self.cv_dim:
            norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
            norm_coord_imgs_2d[..., 2] = 0
            Voxel_2D = F.grid_sample(RPN_feature.unsqueeze(2), norm_coord_imgs_2d, align_corners=True)
            # Voxel_2D = build_geometry_volume(RPN_feature, norm_coord_imgs_2d[..., :2])
        else:
            voxel_disps_channels = self.compute_disp_channels(voxel_disps, 
                img_channels=RPN_feature.shape[1], inv_ratio=self.inv_smooth_geo)
            Voxel_2D = build_dps_geometry_volume(RPN_feature, norm_coord_imgs[..., :2], \
                voxel_disps_channels.to(torch.int32), self.cv_dim, self.geometry_volume_shift)
        return Voxel_2D

    def build_plane_sweep_volume(self, RPN_feature, norm_coord_imgs, voxel_disps):
        if RPN_feature.shape[1] <= self.cv_dim:
            norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
            norm_coord_imgs_2d[..., 2] = 0
            Voxel_2D = F.grid_sample(RPN_feature.unsqueeze(2), norm_coord_imgs_2d, align_corners=True)
            # Voxel_2D = build_geometry_volume(RPN_feature, norm_coord_imgs_2d[..., :2])
        else:
            voxel_disps_channels = self.compute_disp_channels(voxel_disps, 
                img_channels=RPN_feature.shape[1])
            Voxel_2D = build_dps_geometry_volume(RPN_feature, norm_coord_imgs[..., :2], \
                voxel_disps_channels.to(torch.int32), 32, self.geometry_volume_shift)
        return Voxel_2D

    def forward(self, batch_dict):
        left = batch_dict['left_img']
        calib = batch_dict['calib']
        fu_mul_baseline = torch.as_tensor(
            [x.fu_mul_baseline for x in calib], dtype=torch.float32, device=left.device)
        calibs_Proj = torch.as_tensor(
            [x.P2 for x in calib], dtype=torch.float32, device=left.device)
        calibs_Proj_R = torch.as_tensor(
            [x.P3 for x in calib], dtype=torch.float32, device=left.device)

        N = batch_dict['batch_size']

        # feature extraction
        left_features = self.feature_backbone(left)
        left_features = [left] + list(left_features)
        left_stereo_feat, left_sem_feat = self.feature_neck(left_features)

        if not self.mono:
            right = batch_dict['right_img']
            right_features = self.feature_backbone(right)
            right_features = [right] + list(right_features)
            right_stereo_feat, right_sem_feat = self.feature_neck(right_features)
        else:
            right_stereo_feat, right_sem_feat = None, None

        if self.sem_neck is not None:
            batch_dict['sem_features'] = self.sem_neck([left_sem_feat])
        else:
            batch_dict['sem_features'] = [left_sem_feat]
        batch_dict['left_rpn_feature'] = left_sem_feat
        if not self.mono:
            batch_dict['right_rpn_feature'] = right_sem_feat

        if not self.drop_psv:
            # stereo matching: build stereo volume
            downsampled_depth = self.downsampled_depth.cuda()
            downsampled_disp = fu_mul_baseline[:, None] / \
                downsampled_depth[None, :] / (self.downsample_disp if not self.fullres_stereo_feature else 1)

            if left_stereo_feat.shape[1] > self.cv_dim:
                psv_disps_channels = self.compute_disp_channels(downsampled_disp[0], left_stereo_feat.shape[1], inv_ratio=self.inv_smooth_psv)
                cost_raw = self.build_cost(left_stereo_feat, right_stereo_feat,
                                        None, None, downsampled_disp, psv_disps_channels.to(torch.int32))
            else:
                cost_raw = self.build_cost(left_stereo_feat, right_stereo_feat,
                                        None, None, downsampled_disp)

            # stereo matching network
            cost0 = self.dres0(cost_raw)
            cost0 = self.dres1(cost0) + cost0
            if len(self.hg_stereo) > 0:
                all_costs = []
                cur_cost = cost0
                assert len(self.hg_stereo) == 1
                for hg_stereo_module in self.hg_stereo:
                    cost_residual = hg_stereo_module(cur_cost, None, None)
                    cur_cost = cur_cost + cost_residual
                    all_costs.append(cur_cost)
            else:
                all_costs = [cost0]
            assert len(all_costs) > 0, 'at least one hourglass'

            if not self.drop_psv_loss:
                # stereo matching: outputs
                batch_dict['depth_preds'] = []
                if not self.training:
                    batch_dict['depth_preds_local'] = []
                batch_dict['depth_volumes'] = []
                batch_dict['depth_samples'] = self.depth.clone().detach().cuda()
                for idx in range(len(all_costs)):
                    upcost_i, cost_softmax_i, pred_i = self.pred_depth(self.pred_stereo[idx], all_costs[idx], left.shape[2:4])
                    batch_dict['depth_volumes'].append(upcost_i)
                    batch_dict['depth_preds'].append(pred_i)
                    if not self.training:
                        batch_dict['depth_preds_local'].append(self.get_local_depth(cost_softmax_i))
                
            # beginning of 3d detection part
            if self.use_stereo_out_type == "feature":
                out = all_costs[-1]
            elif self.use_stereo_out_type == "prob":
                out = cost_softmax_i.unsqueeze(1)
            elif self.use_stereo_out_type == "cost":
                out = upcost_i.unsqueeze(1)
            else:
                raise ValueError('wrong self.use_stereo_out_type option')
            
        # convert plane-sweep into 3d volume
        coordinates_3d = self.coordinates_3d.cuda()
        batch_dict['coord'] = coordinates_3d
        norm_coord_imgs = []
        if self.cat_right_img_feature:
            norm_coord_imgs_R = []
        valids2d = []
        for i in range(N):
            # map to rect camera coordinates
            c3d = coordinates_3d.view(-1, 3)
            if 'random_T' in batch_dict:
                random_T = batch_dict['random_T'][i]
                c3d = torch.matmul(c3d, random_T[:3, :3].T) + random_T[:3, 3]

            # in pseudo lidar coord
            c3d = project_pseudo_lidar_to_rectcam(c3d)
            #------------ left images ----------------------
            coord_img, norm_coord_img = self.compute_mapping(c3d,
                left.shape[2:],
                torch.as_tensor(calib[i].P2, device='cuda', dtype=torch.float32),
                [self.CV_DEPTH_MIN, self.CV_DEPTH_MAX])
            coord_img = coord_img.view(*self.coordinates_3d.shape[:3], 3)
            norm_coord_img = norm_coord_img.view(*self.coordinates_3d.shape[:3], 3)
            norm_coord_imgs.append(norm_coord_img)

            if self.cat_right_img_feature:
                #------------ right images ----------------------
                coord_img_R, norm_coord_img_R = self.compute_mapping(c3d,
                    right.shape[2:],
                    torch.as_tensor(calib[i].P3, device='cuda', dtype=torch.float32),
                    [self.CV_DEPTH_MIN, self.CV_DEPTH_MAX])
                coord_img_R = coord_img_R.view(*self.coordinates_3d.shape[:3], 3)
                norm_coord_img_R = norm_coord_img_R.view(*self.coordinates_3d.shape[:3], 3)
                norm_coord_imgs_R.append(norm_coord_img_R)

            # valid: within images
            img_shape = batch_dict['image_shape'][i]
            valid_mask_2d = (coord_img[..., 0] >= 0) & (coord_img[..., 0] <= img_shape[1]) & \
                (coord_img[..., 1] >= 0) & (coord_img[..., 1] <= img_shape[0])
            valids2d.append(valid_mask_2d)

        norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
        if self.cat_right_img_feature:
            norm_coord_imgs_R = torch.stack(norm_coord_imgs_R, dim=0)
        valids2d = torch.stack(valids2d, dim=0)
        batch_dict['norm_coord_imgs'] = norm_coord_imgs

        valids = valids2d & (norm_coord_imgs[..., 2] >= -1.) & (norm_coord_imgs[..., 2] <= 1.)
        batch_dict['valids'] = valids
        valids = valids.float()

        if not self.drop_psv:
            # Retrieve Voxel Feature from Cost Volume Feature
            Voxel = F.grid_sample(out, norm_coord_imgs, align_corners=True)
            Voxel = Voxel * valids[:, None, :, :, :]
            Voxels = [Voxel]
        else:
            Voxels = []

        voxel_depths = c3d.view(coordinates_3d.shape)[0,0,:,2]
        voxel_disps = calib[0].fu_mul_baseline / voxel_depths

        # Retrieve Voxel Feature from 2D Img Feature
        if self.cat_img_feature:
            Voxel_2D = self.build_3d_geometry_volume(left_sem_feat, norm_coord_imgs, voxel_disps)
            Voxel_2D *= valids2d.float()[:, None, :, :, :]
            Voxels.append(Voxel_2D)

        if self.cat_right_img_feature:
            Voxel_2D_R = self.build_3d_geometry_volume(right_sem_feat, norm_coord_imgs_R, voxel_disps)
            Voxel_2D_R *= valids2d.float()[:, None, :, :, :]
            Voxels.append(Voxel_2D_R)

        if self.squeeze_geo:
            Voxel = self.squeeze_geo_conv(torch.cat([Voxels[-2], Voxels[-1]], dim=1) if self.cat_right_img_feature else Voxels[-1] )
            if not self.drop_psv:
                Voxel = torch.cat([Voxels[0], Voxel], dim=1)
        else:
            Voxel = Voxels[0] if len(Voxels) == 1 else torch.cat(Voxels, dim=1)

        Voxel = self.rpn3d_convs(Voxel) 
        if self.num_3dconvs_hg > 0:
            if self.num_3dconvs_hg == 1:
                pre, post = True, True
                for hg_stereo_module in self.rpn3d_hgs:
                    Voxel, pre, post = hg_stereo_module(Voxel, pre, post)
            else:
                pre, post = None, None
                for hg_stereo_module in self.rpn3d_hgs:
                    Voxel = hg_stereo_module(Voxel, pre, post)
        batch_dict['volume_features_nopool'] = Voxel

        if self.sup_geometry == 'volume':
            Voxel_for_geo = Voxel

        Voxel = self.rpn3d_pool(Voxel) 

        if self.sup_geometry == 'pooledvolume':
            Voxel_for_geo = Voxel
        batch_dict['volume_features'] = Voxel

        if self.voxel_occupancy:
            batch_dict = self.forward_voxel_occupancy(batch_dict, Voxel_for_geo)

        if self.front_surface_depth:
            batch_dict = self.forward_front_surface_depth_head(batch_dict, Voxel_for_geo, calibs_Proj)

        return batch_dict

    def forward_voxel_occupancy(self, batch_dict, Voxel):
        VoxelOccupancy = self.pred_voxel(self.pred_occupancy, Voxel)
        batch_dict['voxel_occupancy'] = VoxelOccupancy

        return batch_dict

    def forward_front_surface_depth_head(self, batch_dict, Voxel, calibs_Proj):
        coordinates_psv = self.coordinates_psv.cuda()
        dim1, dim2, dim3, _ = coordinates_psv.shape
        coordinates_psv = coordinates_psv.view(-1, 3)
        N = len(calibs_Proj)

        coordinates_psv_to_pseudo_lidars = []
        for i in range(N):
            coordinates_psv_to_pseudo_lidar = unproject_image_to_pseudo_lidar(coordinates_psv, calibs_Proj[i].float().cuda())
            if 'random_T' in batch_dict:
                inv_random_T = batch_dict['inv_random_T'][i]
                coordinates_psv_to_pseudo_lidar = torch.matmul(coordinates_psv_to_pseudo_lidar, inv_random_T[:3, :3].T) + inv_random_T[:3, 3]
            coordinates_psv_to_pseudo_lidars.append(coordinates_psv_to_pseudo_lidar)
        coordinates_psv_to_pseudo_lidars = torch.stack(coordinates_psv_to_pseudo_lidars, axis=0)
            
        norm_coordinates_psv_to_3d = (coordinates_psv_to_pseudo_lidars - torch.as_tensor([self.X_MIN, self.Y_MIN, self.Z_MIN], device=coordinates_psv_to_pseudo_lidars.device)) / torch.as_tensor(
            [self.X_MAX - self.X_MIN, self.Y_MAX - self.Y_MIN, self.Z_MAX - self.Z_MIN], device=coordinates_psv_to_pseudo_lidars.device)
        norm_coordinates_psv_to_3d = norm_coordinates_psv_to_3d * 2 - 1.
        norm_coordinates_psv_to_3d = norm_coordinates_psv_to_3d.view(N, dim1, dim2, dim3, 3)

        PSV_from_3dgv = F.grid_sample(Voxel, norm_coordinates_psv_to_3d)

        batch_dict['depth_preds'] = []
        if not self.training:
            batch_dict['depth_preds_local'] = []
        batch_dict['depth_volumes'] = []
        batch_dict['depth_samples'] = self.depth.clone().detach().cuda()
        upcost_i, cost_softmax_i, pred_i = self.pred_depth(self.pred_stereo[0], PSV_from_3dgv, batch_dict['left_img'].shape[2:])
        batch_dict['depth_volumes'].append(upcost_i)
        batch_dict['depth_preds'].append(pred_i)
        if not self.training:
            batch_dict['depth_preds_local'].append(self.get_local_depth(cost_softmax_i))

        return batch_dict
