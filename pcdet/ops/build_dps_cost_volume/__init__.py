import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pcdet.ops.build_dps_cost_volume import build_dps_cost_volume_cuda


class _BuildDpsCostVolume(Function):
    @staticmethod
    def forward(ctx, left, right, shift, psv_channels, downsample, sep=32, interval=1):
        ctx.save_for_backward(shift, psv_channels)
        ctx.downsample = downsample
        ctx.channels = left.shape[1]
        ctx.sep = sep
        ctx.interval = interval
        assert torch.all(shift >= 0.)
        output = build_dps_cost_volume_cuda.build_dps_cost_volume_forward(
            left, right, shift, psv_channels, downsample, sep, interval)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        shift, psv_channels = ctx.saved_tensors
        grad_left, grad_right = build_dps_cost_volume_cuda.build_dps_cost_volume_backward(
            grad_output, shift, psv_channels, ctx.downsample, ctx.channels, ctx.sep, ctx.interval)
        return grad_left, grad_right, None, None, None, None, None


build_dps_cost_volume = _BuildDpsCostVolume.apply
