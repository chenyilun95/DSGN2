#include <torch/extension.h>

at::Tensor BuildDpsGeometryVolume_forward_cuda(const at::Tensor &img,
                                        const at::Tensor &coord,
                                        const at::Tensor &disp_channels,
                                        const int sep,
                                        const int interval);

at::Tensor BuildDpsGeometryVolume_backward_cuda(const at::Tensor &grad,
                                                                 const at::Tensor &coord,
                                                                 const at::Tensor &disp_channels,
                                                                 const int height,
                                                                 const int width,
                                                                 const int channels,
                                                                 const int sep,
                                                                 const int interval);

// Interface for Python
at::Tensor BuildDpsGeometryVolume_forward(const at::Tensor &img,
                                   const at::Tensor &coord,
                                   const at::Tensor &disp_channels,
                                   const int sep,
                                   const int interval)
{
  if (img.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildDpsGeometryVolume_forward_cuda(img, coord, disp_channels, sep, interval);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor BuildDpsGeometryVolume_backward(const at::Tensor &grad,
                                                            const at::Tensor &coord,
                                                            const at::Tensor &disp_channels,
                                                            const int height,
                                                            const int width,
                                                            const int channels,
                                                            const int sep,
                                                            const int interval)
{
  if (grad.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildDpsGeometryVolume_backward_cuda(grad, coord, disp_channels, height, width, channels, sep, interval);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("build_dps_geometry_volume_forward", &BuildDpsGeometryVolume_forward, "BuildDpsGeometryVolume_forward");
  m.def("build_dps_geometry_volume_backward", &BuildDpsGeometryVolume_backward, "BuildDpsGeometryVolume_backward");
}
