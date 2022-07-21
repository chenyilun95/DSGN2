import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pcdet.ops.build_dps_geometry_volume import build_dps_geometry_volume_cuda


class _BuildDpsGeometryVvolume(Function):
    @staticmethod
    def forward(ctx, img, coord, disp_channels, sep, interval):
        ctx.save_for_backward(coord, disp_channels)
        ctx.channels = img.shape[1]
        ctx.img_height = img.shape[2]
        ctx.img_width = img.shape[3]
        ctx.sep = sep
        ctx.interval = interval
        output = build_dps_geometry_volume_cuda.build_dps_geometry_volume_forward(
            img, coord, disp_channels, sep, interval
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        coord, disp_channels = ctx.saved_tensors
        grad_input = build_dps_geometry_volume_cuda.build_dps_geometry_volume_backward(
            grad_output,
            coord,
            disp_channels,
            ctx.img_height,
            ctx.img_width,
            ctx.channels,
            ctx.sep,
            ctx.interval
        )
        return grad_input, None, None, None, None


build_dps_geometry_volume = _BuildDpsGeometryVvolume.apply
