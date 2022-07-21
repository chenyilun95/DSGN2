// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width,
    T y, T x) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T> 
__global__ void BuildGeometryVolumeForward(const int nthreads, 
    const T* img, const T* coord, 
    const int num_batch, const int channels, const int height,
    const int width, const int z_num, const int y_num, const int x_num,
    T* volume) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % x_num;
    int ph = (index / x_num) % y_num;
    int pd = (index / x_num / y_num) % z_num;
    int c = (index / x_num / y_num/ z_num) % channels;
    int n = index / x_num / y_num / z_num / channels;

    int index_coord_x = (((n * z_num + pd) * y_num + ph) * x_num + pw) * 2;
    int index_coord_y = index_coord_x + 1;
    T coord_y = (coord[index_coord_y] + 1.) / 2. * (height - 1.);
    T coord_x = (coord[index_coord_x] + 1.) / 2. * (width - 1.);

    const T* offset_input = img + (n * channels + c) * height * width;
    volume[index] = bilinear_interpolate(offset_input, height, width, coord_y, coord_x);
  }
}


template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void BuildGeometryVolumeBackwardFeature(const int nthreads, 
    const T* grad, const T* coord, 
    const int num_batch, const int channels, const int height,
    const int width, const int z_num, const int y_num, const int x_num,
    T* grad_input) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % x_num;
    int ph = (index / x_num) % y_num;
    int pd = (index / x_num / y_num) % z_num;
    int c = (index / x_num / y_num/ z_num) % channels;
    int n = index / x_num / y_num / z_num / channels;

    int index_coord_x = (((n * z_num + pd) * y_num + ph) * x_num + pw) * 2;
    int index_coord_y = index_coord_x + 1;
    T coord_y = (coord[index_coord_y] + 1.) / 2. * (height - 1.);
    T coord_x = (coord[index_coord_x] + 1.) / 2. * (width - 1.);

    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    bilinear_interpolate_gradient(height, width, coord_y, coord_x,
        w1, w2, w3, w4,
        x_low, x_high, y_low, y_high);

    T top_diff_this_bin = grad[index];
    T g1 = top_diff_this_bin * w1;
    T g2 = top_diff_this_bin * w2;
    T g3 = top_diff_this_bin * w3;
    T g4 = top_diff_this_bin * w4;

    T* offset_grad_input = grad_input + (n * channels + c) * height * width;
    if (w1 >= 1e-10)
        atomicAdd(offset_grad_input + y_low * width + x_low, static_cast<T>(g1));
    if (w2 >= 1e-10)
        atomicAdd(offset_grad_input + y_low * width + x_high, static_cast<T>(g2));
    if (w3 >= 1e-10)
        atomicAdd(offset_grad_input + y_high * width + x_low, static_cast<T>(g3));
    if (w4 >= 1e-10)
        atomicAdd(offset_grad_input + y_high * width + x_high, static_cast<T>(g4));
  } // CUDA_1D_KERNEL_LOOP
} // BuildGeometryVolumeBackward


at::Tensor BuildGeometryVolume_forward_cuda(const at::Tensor& img,
                                 const at::Tensor& coord) {
  AT_ASSERTM(img.type().is_cuda(), "img must be a CUDA tensor");
  AT_ASSERTM(coord.type().is_cuda(), "coord must be a CUDA tensor");

  AT_ASSERTM((img.size(0) == coord.size(0)) && (coord.size(4) == 2), \
    "Image and coord should of same batch.");

  auto num_batch = img.size(0);
  auto channels = img.size(1);
  auto height = img.size(2);
  auto width = img.size(3);
  auto z_num = coord.size(1);
  auto y_num = coord.size(2);
  auto x_num = coord.size(3);

  auto output = at::empty({num_batch, channels, z_num, y_num, x_num}, img.options());
  auto output_size = num_batch * channels * z_num * y_num * x_num;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)(output_size), 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(img.type(), "BuildGeometryVolume_forward", [&] {
    BuildGeometryVolumeForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         img.contiguous().data<scalar_t>(),
         coord.contiguous().data<scalar_t>(),
         num_batch,
         channels,
         height,
         width,
         z_num,
         y_num,
         x_num,
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor BuildGeometryVolume_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& coord,
                                  const int height,
                                  const int width) {
  AT_ASSERTM(coord.type().is_cuda(), "coord must be a CUDA tensor");

  auto num_batch = grad.size(0);
  auto channels = grad.size(1);
  auto z_num = grad.size(2);
  auto y_num = grad.size(3);
  auto x_num = grad.size(4);
  
  auto grad_input = at::zeros({num_batch, channels, height, width}, grad.options());

  AT_ASSERTM((z_num == coord.size(1)) && (y_num == coord.size(2)) && (x_num == coord.size(3)),
      "grad shape is wrong");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "BuildGeometryVolume_backward", [&] {
    BuildGeometryVolumeBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         coord.contiguous().data<scalar_t>(),
         num_batch,
         channels,
         height,
         width,
         z_num,
         y_num,
         x_num,
         grad_input.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}

