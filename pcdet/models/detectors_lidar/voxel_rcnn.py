from .lidar_detector3d_template import Detector3DTemplate


class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.model_cfg.get('RETURN_BATCH_DICT', False):
            # keys_to_transfer = ['multi_scale_3d_strides', 
            #                       'multi_scale_3d_features']
            keys_to_remove = ['reg_features',
                              'box_cls_labels',
                              'box_reg_targets',
                              'reg_weights',
                              'anchors',
                              'cls_preds_normalized']
            if not self.model_cfg.get('RETURN_SPARSE_DICT', False):
                keys_to_remove.extend(['encoded_spconv_tensor',
                              'encoded_spconv_tensor_stride',])
            # 'voxels', 'voxel_coords', 'voxel_num_points', 'voxel_features']
            for k in keys_to_remove:
                batch_dict.pop(k, None)
            
            # 'spatial_features_2d_prehg',
            keys_to_keep = ['spatial_features_stride',
                            'spatial_features',
                            'spatial_features_2d',
                            'volume_features',
                            'batch_cls_preds',
                            'batch_box_preds']
            if self.model_cfg.get('RETURN_SPARSE_DICT', False):
                keys_to_keep.extend(['encoded_spconv_tensor',
                            'encoded_spconv_tensor_stride',
                            'voxel_coords'])
            batch_dict['lidar_outputs'] = {}
            for k in keys_to_keep:
                if k in batch_dict:
                    batch_dict['lidar_outputs'][k] = batch_dict.pop(k)
            
            return batch_dict

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
