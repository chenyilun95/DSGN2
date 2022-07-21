import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources, define_macros=None, extra_compile_args=None):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
    return cuda_ext


if __name__ == '__main__':
    version = '0.1.0+%s' % get_git_commit_number()

    setup(
        name='pcdet',
        version=version,
        description='Deep Stereo Geometry Network',
        install_requires=[
            'numpy',
            'torch>=1.1',
            'spconv',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='build_cost_volume_cuda',
                module='pcdet.ops.build_cost_volume',
                sources=[
                    'src/BuildCostVolume.cpp',
                    'src/BuildCostVolume_cuda.cu',
                ],
                define_macros=[("WITH_CUDA", None)]
            ),
            make_cuda_ext(
                name='build_dps_cost_volume_cuda',
                module='pcdet.ops.build_dps_cost_volume',
                sources=[
                    'src/BuildDpsCostVolume.cpp',
                    'src/BuildDpsCostVolume_cuda.cu',
                ],
                define_macros=[("WITH_CUDA", None)]
            ),
            make_cuda_ext(
                name='build_geometry_volume_cuda',
                module='pcdet.ops.build_geometry_volume',
                sources=[
                    'src/BuildGeometryVolume.cpp',
                    'src/BuildGeometryVolume_cuda.cu',
                ],
                define_macros=[("WITH_CUDA", None)]
            ),
            make_cuda_ext(
                name='build_dps_geometry_volume_cuda',
                module='pcdet.ops.build_dps_geometry_volume',
                sources=[
                    'src/BuildDpsGeometryVolume.cpp',
                    'src/BuildDpsGeometryVolume_cuda.cu',
                ],
                define_macros=[("WITH_CUDA", None)]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp', 
                    'src/voxel_query_gpu.cu',
                    'src/vector_pool.cpp',
                    'src/vector_pool_gpu.cu'
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),
        ],
    )
