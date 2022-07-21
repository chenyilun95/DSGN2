# Stereo cost volume builder.


import torch
from torch import nn

from pcdet.ops.build_cost_volume import build_cost_volume
from pcdet.ops.build_dps_cost_volume import build_dps_cost_volume

class BuildCostVolume(nn.Module):
    def __init__(self, volume_cfgs):
        self.volume_cfgs = volume_cfgs
        super(BuildCostVolume, self).__init__()

    def get_dim(self, feature_channel):
        d = 0
        for cfg in self.volume_cfgs:
            volume_type = cfg["type"]
            if volume_type == "concat":
                d += 32 * 2
        return d

    def forward(self, left, right, left_raw, right_raw, shift, psv_disps_channels=None):
        volumes = []
        for cfg in self.volume_cfgs:
            volume_type = cfg["type"]

            if volume_type == "concat":
                downsample = getattr(cfg, "downsample", 1)
                if left.shape[1] == 32:
                    volumes.append(build_cost_volume(left, right, shift, downsample))
                else:
                    volumes.append(build_dps_cost_volume(left, right, shift, psv_disps_channels, downsample, 32, getattr(cfg, "shift", 1)))
            else:
                raise NotImplementedError
        if len(volumes) > 1:
            return torch.cat(volumes, dim=1)
        else:
            return volumes[0]

    def __repr__(self):
        tmpstr = self.__class__.__name__
        return tmpstr
