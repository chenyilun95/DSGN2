# Depth Loss Head for stereo matching supervision.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from pcdet.utils.loss_utils import sigmoid_focal_loss

class VoxelLossHead(nn.Module):
    def __init__(self, model_cfg, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_loss_type = model_cfg.LOSS_TYPE
        self.loss_weights = model_cfg.WEIGHTS
        self.point_cloud_range = point_cloud_range
        self.min_depth = point_cloud_range[0]
        self.max_depth = point_cloud_range[3]
        self.pos_weight = model_cfg.POS_WEIGHT

    def get_loss(self, batch_dict, tb_dict=None):     
        if tb_dict is None:
            tb_dict = {}

        pred_voxel_occupancy = batch_dict['voxel_occupancy'].squeeze(1)

        N = pred_voxel_occupancy.shape[0]

        voxel_loss = 0.
        for i, (voxels_in_ray, occupany_of_voxels_in_ray, pred_voxel_occupancy_in_ray) in enumerate(zip(
                batch_dict['voxels_in_ray'], batch_dict['occupany_of_voxels_in_ray'],
                pred_voxel_occupancy)):
            
            if len(voxels_in_ray) == 0: # none depth
                voxel_loss += pred_voxel_occupancy.sum() * 0.
                continue
            
            voxels_in_ray = torch.as_tensor(voxels_in_ray, device='cuda')
            occupany_of_voxels_in_ray = torch.as_tensor(occupany_of_voxels_in_ray, device='cuda')
            pred_voxel_occupancy_in_ray = pred_voxel_occupancy_in_ray.view(-1)[voxels_in_ray]

            if self.voxel_loss_type == 'bce_loss':
                voxel_loss_per_batch = F.binary_cross_entropy_with_logits(pred_voxel_occupancy_in_ray, occupany_of_voxels_in_ray, reduction='none')
            elif self.voxel_loss_type == 'sigmoid_focal_loss':
                voxel_loss_per_batch = sigmoid_focal_loss(pred_voxel_occupancy_in_ray, occupany_of_voxels_in_ray, reduction='none')

            # weight
            weight = torch.ones_like(voxel_loss_per_batch)
            if self.pos_weight:
                norm_dist = torch.as_tensor(batch_dict['norm_dist'][i], device='cuda')
                weight *= norm_dist
            voxel_loss_per_batch = (voxel_loss_per_batch * weight).sum() / (occupany_of_voxels_in_ray * weight).sum()
            voxel_loss += voxel_loss_per_batch / N
        
        voxel_loss = voxel_loss * self.loss_weights
        return voxel_loss, tb_dict

    def forward(self, batch_dict):
        return batch_dict
