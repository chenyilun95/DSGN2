"""a series of useful pytorch operations related to bbox transformation."""
import torch
import numpy as np 


def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32 

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    else:
        raise ValueError('ndim shoule be 2 or 3')
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    # angles: [N]
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, zeros, -rot_sin]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, -rot_sin, zeros]),
            torch.stack([rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        # TODO: check why not in stand form?
        rot_mat_T = torch.stack([
            torch.stack([zeros, rot_cos, -rot_sin]),
            torch.stack([zeros, rot_sin, rot_cos]),
            torch.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should in range")

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def center_to_corner_box3d(centers,
                           dims,
                           angles,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.view(-1, 1, 3)
    return corners


def lidar_to_camera(points, r_rect, velo2cam):
    """
    TODO: how to handle any dimensional points data

    :param points: [B, N, 3/4]
    :param r_rect: [B, 4, 4]
    :param velo2cam: [B, 4, 4]
    :return:
    """
    assert len(points.shape) == 2 or len(points.shape) == 3
    assert len(points.shape) == len(r_rect.shape)
    assert len(points.shape) == len(velo2cam.shape)
    points = torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)
    camera_points = points @ (r_rect @ velo2cam).t()
    return camera_points[..., :3]


def pseudo_lidar_to_camera(points):
    assert points.shape[-1] == 3
    points_shape = points.shape[:-1]
    r_rect = torch.eye(4, dtype=torch.float32, device=points.device)
    velo2cam = torch.tensor([[0, -1, 0, 0],  # cam x: -velo y
                         [0, 0, -1, 0],  # cam y: -velo z
                         [1, 0, 0, 0],  # cam z: velo x
                         [0, 0, 0, 1]], dtype=torch.float32, device=points.device)
    return lidar_to_camera(points.view(-1, 3), r_rect, velo2cam).view(*points_shape, 3)


def camera_to_lidar(points, r_rect, velo2cam):
    """transform points in camera coordinate to lidar coordinate

    :param points: [B, ..., 3]  points in camera coordinate
    :param r_rect: [B, 4, 4] camera rectification transformation matrix
    :param velo2cam: [B, 4, 4] velo to cam transformation matrix
    :return: lidar_points: [B, ..., 3]  points in LIDAR coordinate
    """
    points = torch.cat([points, torch.ones_like(points[..., 0:1])], dim=-1)
    assert points.shape[-1] == 4
    shape_per_sample = points.shape[1:-1]
    batch_size = points.shape[0]
    points = points.view(batch_size, -1, 4)
    lidar_points = points @ torch.inverse(r_rect @ velo2cam).transpose(-2, -1)
    lidar_points = lidar_points.view(batch_size, *shape_per_sample, 4)
    return lidar_points[..., :3]


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return torch.cat([xyz, l, h, w, r], dim=-1)


def box_pseudo_lidar_to_camera(data):
    xyz_lidar = data[..., 0:3]
    w, l, h = data[..., 3:4], data[..., 4:5], data[..., 5:6]
    r = data[..., 6:7]
    xyz = pseudo_lidar_to_camera(xyz_lidar)
    return torch.cat([xyz, l, h, w, r], dim=-1)


def project_to_image(points_3d, proj_mat):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    points_4 = torch.cat(
        [points_3d, torch.zeros(*points_shape).type_as(points_3d)], dim=-1)
    # point_2d = points_4 @ tf.transpose(proj_mat, [1, 0])
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res
