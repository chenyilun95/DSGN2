import torch
import torchvision
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class CCNN(nn.Module):    
    def __init__(self, model_cfg):
        super(CCNN, self).__init__()
        self.model_cfg = model_cfg

        kernel_size = 3
        filters = self.model_cfg.FC_FILTERS
        fc_filters = 128
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            # nn.Conv2d(288, 124, kernel_size, stride=1, padding=padding),
            # nn.ReLU(),
            # nn.Conv2d(288, filters, kernel_size, stride=1, padding=padding),
            nn.Conv2d(1, filters, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(288, fc_filters, 1),
            nn.ReLU(),
            nn.Conv2d(fc_filters, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(fc_filters, 1, 1),
        )
        self.activate = nn.Sigmoid()
    
    def gt_depth_confidence_map(self, batch_dict):
        
        batch_cf_map_label = []
        for i in range(batch_dict['batch_size']):
            gt_disparity_map = batch_dict['depth_gt_img'][i].detach().cpu().numpy()
            valid_pixels_mask = gt_disparity_map > 0
            depth_preds = batch_dict['depth_preds'][i].detach().cpu().numpy()
            depth_preds = batch_dict['calib'][i].depth_to_disparity(depth_preds)
            gt_disparity_map = batch_dict['calib'][i].depth_to_disparity(gt_disparity_map)

            confidence_map_label = torch.tensor(abs(depth_preds * valid_pixels_mask - gt_disparity_map) < 3, dtype=torch.float32)
            batch_cf_map_label.append(confidence_map_label)
        
        confidence_map_label = torch.cat(batch_cf_map_label, dim=0)

        return confidence_map_label, valid_pixels_mask
        
    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']
        for i in range(batch_size):
            depth_volumes = batch_dict["depth_volumes"][i]
            # out = self.conv(input)
            out = self.fc(depth_volumes)
            # out = (out - out.min(dim=1).values) / (out.max(dim=1).values - out.min(dim=1).values)
            out = self.activate(out)

            batch_dict['batch_feature_depth'] = out
            confidence_map_label, valid_pixels_mask = self.gt_depth_confidence_map(batch_dict)
            batch_dict['confidence_map_label'] = confidence_map_label
            batch_dict['valid_pixels_mask'] = valid_pixels_mask
            if self.training:
                self.forward_ret_dict = batch_dict

        return batch_dict
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        dcm_loss = 0
        dcm_loss_cls, cls_tb_dict = self.get_depth_confidence_loss(self.forward_ret_dict)
        dcm_loss += dcm_loss_cls
        tb_dict.update(cls_tb_dict)

        tb_dict['dcm_loss'] = dcm_loss.item()
        return dcm_loss, tb_dict

    def get_depth_confidence_loss(self, forward_ret_dict):
        confidence_map_label = forward_ret_dict['confidence_map_label']
        depth_feature = forward_ret_dict['batch_feature_depth']
        valid_pixels_mask = forward_ret_dict['valid_pixels_mask']
        
        for i in range(forward_ret_dict['batch_size']):
            depth_feature = depth_feature[i]
            loss_func = nn.BCELoss()
            batch_loss_dcm = loss_func(
                depth_feature*torch.tensor(valid_pixels_mask).cuda(), confidence_map_label.cuda()
            )

            tb_dict = {'dcm_loss': batch_loss_dcm.item()}
        return batch_loss_dcm, tb_dict
