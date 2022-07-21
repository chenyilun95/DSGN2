#include <torch/extension.h>

at::Tensor BuildDpsCostVolume_forward_cuda(const at::Tensor &left,
                                        const at::Tensor &right,
                                        const at::Tensor &shift,
                                        const at::Tensor &psv_channels,
                                        const int downsample,
                                        const int sep,
                                        const int interval);

std::tuple<at::Tensor, at::Tensor> BuildDpsCostVolume_backward_cuda(const at::Tensor &grad,
                                                                 const at::Tensor &shift,
                                                                 const at::Tensor &psv_channels,
                                                                 const int downsample,
                                                                 const int channels,
                                                                 const int sep,
                                                                 const int interval);

// Interface for Python
at::Tensor BuildDpsCostVolume_forward(const at::Tensor &left,
                                   const at::Tensor &right,
                                   const at::Tensor &shift,
                                   const at::Tensor &psv_channels,
                                   const int downsample,
                                   const int sep,
                                   const int interval)
{
  if (left.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildDpsCostVolume_forward_cuda(left, right, shift, psv_channels, downsample, sep, interval);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> BuildDpsCostVolume_backward(const at::Tensor &grad,
                                                            const at::Tensor &shift,
                                                            const at::Tensor &psv_channels,
                                                            const int downsample,
                                                            const int channels,
                                                            const int sep,
                                                            const int interval)
{
  if (grad.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildDpsCostVolume_backward_cuda(grad, shift, psv_channels, downsample, channels, sep, interval);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("build_dps_cost_volume_forward", &BuildDpsCostVolume_forward, "BuildDpsCostVolume_forward");
  m.def("build_dps_cost_volume_backward", &BuildDpsCostVolume_backward, "BuildDpsCostVolume_backward");
}
