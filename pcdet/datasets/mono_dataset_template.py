from collections import defaultdict
from pathlib import Path
import numpy as np
import torch.utils.data as torch_data

from pcdet.utils import common_utils, box_utils, depth_map_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .augmentor.mono_data_augmentor import MonoDataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from pcdet.utils.calibration_kitti import Calibration
from .stereo_dataset_template import StereoDatasetTemplate

class MonoDatasetTemplate(StereoDatasetTemplate):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super(MonoDatasetTemplate, self).__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)

        if self.training:
            self.data_augmentor = MonoDataAugmentor(
                self.root_path, self.dataset_cfg.TRAIN_DATA_AUGMENTOR, self.class_names, logger=self.logger
            )
        else:
            if getattr(self.dataset_cfg, 'TEST_DATA_AUGMENTOR', None) is not None:
                self.data_augmentor = MonoDataAugmentor(
                    self.root_path, self.dataset_cfg.TEST_DATA_AUGMENTOR, self.class_names, logger=self.logger
                )
                # logger.warn('using data augmentor in test mode')
            else:
                self.data_augmentor = None

    def prepare_data(self, data_dict):
        data_dict = super(MonoDatasetTemplate, self).prepare_data(data_dict)
        
        data_dict.pop('right_img')

        return data_dict

