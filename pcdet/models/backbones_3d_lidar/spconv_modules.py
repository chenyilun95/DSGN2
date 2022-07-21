from functools import partial

import torch
import torch.nn as nn

import spconv
from spconv.modules import SparseModule

from collections import OrderedDict

from ...utils import common_utils
from .spconv_backbone import post_act_block

def expand_points(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0):
    assert out_channels == in_channels
    assert kernel_size != 1
    assert stride == 1

    conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                            bias=False, indice_key=indice_key)
    conv.weight.requires_grad = False
    conv.weight[:, :, :] = torch.eye(in_channels)
    return conv

class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes
        
        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='hg_subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='hg_spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='hg_spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='hg_subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        return output


class SpHourglass(nn.Module):
    def __init__(self, inplanes, block_reps=1, block_residual=True, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        self.unet = UBlock([inplanes, 2*inplanes, 4*inplanes], norm_fn, block_reps, block, indice_key_id=1)
    
    def forward(self, x):
        return self.unet(x)

