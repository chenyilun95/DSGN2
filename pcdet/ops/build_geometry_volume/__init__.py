import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pcdet.ops.build_geometry_volume import build_geometry_volume_cuda


class _BuildGeometryVolume(Function):
    @staticmethod
    def forward(ctx, img, coord):
        ctx.save_for_backward(coord)
        ctx.img_height = img.shape[2]
        ctx.img_width = img.shape[3]
        output = build_geometry_volume_cuda.build_geometry_volume_forward(
            img, coord
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        coord, = ctx.saved_tensors
        grad_input = build_geometry_volume_cuda.build_geometry_volume_backward(
            grad_output,
            coord,
            ctx.img_height,
            ctx.img_width
        )
        return grad_input, None


build_geometry_volume = _BuildGeometryVolume.apply
