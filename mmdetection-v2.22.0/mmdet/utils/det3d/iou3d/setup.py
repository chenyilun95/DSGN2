import os
os.environ["CC"] = "/usr/bin/gcc-5"
os.environ["CXX"] = "/usr/bin/g++-5"
os.environ["CUDA_HOME"] = "/usr/local/cuda-9.0"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iou3d',
    ext_modules=[
        CUDAExtension('iou3d_cuda', [
            'src/iou3d.cpp',
            'src/iou3d_kernel.cu',
        ],
        # extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda-9.0/include'],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
