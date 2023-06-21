import torch
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .stereo_detector3d_template import StereoDetector3DTemplate

from pcdet.utils.common_utils import T

class STEREO(StereoDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # with T(self.__module__.split('.')[-1], record=True, enable=not self.training):
        #     for cur_module in self.module_list:
        #         with T(cur_module.__module__.split('.')[-1], enable=not self.training):
        #             batch_dict = cur_module(batch_dict)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.model_cfg.get('RETURN_BATCH_DICT', False):
            keys_to_remove = ['sem_features',
                'rpn_feature',
                'valids',
                'norm_coord_imgs',
                'volume_features_nopool',
                ]
            for k in keys_to_remove:
                batch_dict.pop(k, None)
            # 'spatial_features_2d_prehg',
            keys_to_keep = ['spatial_features_stride',
                            'spatial_features',
                            'spatial_features_2d',
                            'volume_features',
                            'batch_cls_preds',
                            'batch_box_preds']
            batch_dict['lidar_outputs'] = {} # teacher outputs
            for k in keys_to_keep:
                if k in batch_dict:
                    batch_dict['lidar_outputs'][k] = batch_dict.pop(k)
            
            return batch_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, ret_dicts = self.post_processing(batch_dict)
            for k in batch_dict.keys():
                if k.startswith('depth_error_'):
                    if isinstance(batch_dict[k], list):
                        ret_dicts[k] = batch_dict[k]
                    elif len(batch_dict[k].shape) == 0:
                        ret_dicts[k] = batch_dict[k].item()

            if getattr(self, 'dense_head_2d', None) and 'boxes_2d_pred' in batch_dict:
                assert len(pred_dicts) == len(batch_dict['boxes_2d_pred'])
                for pred_dict, pred_2d_dict in zip(pred_dicts, batch_dict['boxes_2d_pred']):
                    pred_dict['pred_boxes_2d'] = pred_2d_dict['pred_boxes_2d']
                    pred_dict['pred_scores_2d'] = pred_2d_dict['pred_scores_2d']
                    pred_dict['pred_labels_2d'] = pred_2d_dict['pred_labels_2d']
            pred_dicts[0]['batch_dict'] = batch_dict

            return pred_dicts, ret_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss = 0.
        tb_dict = {}
        if getattr(self, 'dense_head', None):
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss += loss_rpn
            tb_dict.update(loss_rpn = loss_rpn.item())

        if (not self.model_cfg.BACKBONE_3D.get('drop_psv', False) and not self.model_cfg.BACKBONE_3D.get('drop_psv_loss', False)) or self.model_cfg.BACKBONE_3D.get('front_surface_depth', False):
            loss_depth, tb_dict = self.depth_loss_head.get_loss(batch_dict, tb_dict)
            tb_dict.update(loss_depth = loss_depth.item())
            if torch.isnan(loss_depth):
                loss += sum([i.sum() for i in batch_dict['depth_preds']]) * 0.
                print('-------------- NaN depth loss')
            else:
                loss += loss_depth

        if self.model_cfg.get('VOXEL_LOSS_HEAD', None):
            loss_voxel, tb_dict = self.voxel_loss_head.get_loss(batch_dict, tb_dict)
            tb_dict.update(loss_voxel=loss_voxel.item())
            if torch.isnan(loss_voxel):
                loss += batch_dict['voxel_occupancy'].sum() * 0.
                print('-------------- NaN depth loss')
            else:
                loss += loss_voxel

        if self.model_cfg.get('RANGE_LOSS_HEAD', None):
            loss_range, tb_dict = self.range_loss_head.get_loss(batch_dict, tb_dict)
            tb_dict.update(loss_range=loss_range.item())
            if torch.isnan(loss_range):
                loss += (sum([i.sum() for i in batch_dict['range_voxel_occupancy']]) + batch_dict['voxel_occupancy'].sum()) * 0.
                print('-------------- NaN depth loss')
            else:
                loss += loss_range

        if getattr(self, 'dense_head_2d', None):
            loss_rpn_2d, tb_dict = self.dense_head_2d.get_loss(batch_dict, tb_dict)
            tb_dict['loss_rpn2d'] = loss_rpn_2d.item()
            loss += loss_rpn_2d
        
        if self.model_cfg.get('ROI_HEAD', None):
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            tb_dict['loss_rcnn'] = loss_rcnn.item()
            loss += loss_rcnn

        return loss, tb_dict, disp_dict

    def get_iou_map(self, batch_dict):
        batch_size = batch_dict['batch_size']
        iou_map_results = []

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]  # [N_anchors, 7]
            gt_boxes = batch_dict['gt_boxes'][index]

            if gt_boxes.shape[0] <= 0:
                iou_map_results.append(None)
            else:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(
                    box_preds[:, 0:7], gt_boxes[:, 0:7])
                iou_map_results.append(iou3d_roi.detach().cpu().numpy())

        return iou_map_results
