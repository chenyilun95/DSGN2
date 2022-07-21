"""a series of useful numpy operations related to bbox transformation."""
import numpy as np
import numba
import mmdet.utils.det3d.geometry as geometry


def camera_to_lidar(points, r_rect, velo2cam):
    """ transformation from camera coordinate to lidar coordinate

    :param points: non-homo coodinates [..., 3] or homogeneous coordinates [..., 4]
    :param r_rect: [4, 4] camera rectification matrix
    :param velo2cam: [4, 4] transformation from velo to camera
    :return: transformed points
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    """ transformation from lidar coordinate to camera coordinate

    :param points: non-homo coodinates [..., 3] or homogeneous coordinates [..., 4]
    :param r_rect: [4, 4] camera rectification matrix
    :param velo2cam: [4, 4] transformation from velo to camera
    :return: transformed points
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ ((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    """ bounding box transformation from camera to lidar

    :param data: [N, 7]
    :param r_rect: [4, 4]
    :param velo2cam: [4, 4]
    :return: transformed bbox
    """
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]  # length, height, width
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def camera_to_pseudo_lidar(points):
    """ transformation from camera coordinate to pseudo lidar coordinate

    :param points: non-homo coodinates [..., 3] or homogeneous coordinates [..., 4]
    :return: transformed points
    """
    r_rect = np.eye(4, dtype=np.float32)
    velo2cam = np.array([[0, -1, 0, 0],  # cam x: -velo y
                         [0, 0, -1, 0],  # cam y: -velo z
                         [1, 0, 0, 0],  # cam z: velo x
                         [0, 0, 0, 1]], dtype=np.float32)
    return camera_to_lidar(points, r_rect, velo2cam)


def pseudo_lidar_to_camera(points):
    """ transformation from  pseudo lidar coordinate to camera coordinate

    :param points: non-homo coodinates [..., 3] or homogeneous coordinates [..., 4]
    :return: transformed points
    """
    r_rect = np.eye(4, dtype=np.float32)
    velo2cam = np.array([[0, -1, 0, 0],  # cam x: -velo y
                         [0, 0, -1, 0],  # cam y: -velo z
                         [1, 0, 0, 0],  # cam z: velo x
                         [0, 0, 0, 1]], dtype=np.float32)
    return lidar_to_camera(points, r_rect, velo2cam)


def box_camera_to_psuedo_lidar(data):
    """ bounding box transformation from camera coordinate to pseudo lidar order

    Why should we do this? Since a lot of data processing functions are order-sensitive. Most of the
    functions take data in lidar coordinate system. However, our stereo framework do not use any data in
    the form of lidar coordinate system. To better utilize these function, we tranform the data in camera

    :param data: [N, 7]
    :return: transformed bbox
    """
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]  # length, height, width
    r = data[:, 6:7]
    xyz_lidar = camera_to_pseudo_lidar(xyz)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def box_psuedo_lidar_to_camera(data):
    """ bounding box transformation from camera coordinate to pseudo lidar order

    Why should we do this? Since a lot of data processing functions are order-sensitive. Most of the
    functions take data in lidar coordinate system. However, our stereo framework do not use any data in
    the form of lidar coordinate system. To better utilize these function, we tranform the data in camera

    :param data: [N, 7]
    :return: transformed bbox
    """
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]  # length, height, width
    r = data[:, 6:7]
    xyz = pseudo_lidar_to_camera(xyz_lidar)
    return np.concatenate([xyz, l, h, w, r], axis=1)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and origin point (0-1 relative encoding).

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point `0-1 relative encoding` relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
    """
    ndim = int(dims.shape[1])
    # corners_norm  [2**ndim, ndim]
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(
        dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        # 00->01->11->10
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        # 000->001->011->010->100->101->111->110
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - \
        np.array(origin, dtype=dims.dtype)  # minus origin encoding
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])  # multiply dims (length) of each dimension
    return corners


def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    batched 2d clockwise rotation for 2d points.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle (radian).

    Returns:
        float array: same shape as points
    """
    rot_sin = np.sin(angles)  # [N]
    rot_cos = np.cos(angles)  # [N]
    rot_mat_T = np.stack(
        [[rot_cos, -rot_sin], [rot_sin, rot_cos]])  # [2, 2, N]
    # batched 2d clockwise rotation for 2d points
    return np.einsum('aij,jka->aik', points, rot_mat_T)  # [N, point_size, 2]


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:  # rotate along y
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:  # rotate along z
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:  # rotate along x
        # TODO: check, why exchange x, y, z axis?
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
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
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    """ box2d to corner coordinates, optimized version for jit

    :param boxes: shape [N, 5]
    :return: corner coordinates: shape [N, 4, 2]
    """
    num_box = boxes.shape[0]
    # 00->01->11->10
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(
        1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    """convert corner coordinates to the ranges of all dimensions.

    :param boxes_corner: [N, 2**ndim, ndim]
    :return: range results: [N, 2*ndim] For each box sample, compute dim0-min, dim1-min, ..., dim0-max, dim1-max, ...
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]):
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7,
                             4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def rotation_points_single_angle(points, angle, axis=0, return_rot=False):
    """rotation points with a single angle

    :param points: [N, 3]
    :param angle: [1]
    :param axis: rotation axis
    :return: rotated points
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype)
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype)
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype)
    else:
        raise ValueError("axis should in range")

    if return_rot:
        return points @ rot_mat_T, rot_mat_T
    else:
        return points @ rot_mat_T


def project_to_image(points_3d, proj_mat):
    """project 3d points from camera coordinate to image coordinate"""
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.zeros(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def box3d_to_bbox(box3d, rect, Trv2c, P2):
    """3d bounding box to 2d bounding box

    :param box3d: [N, 3]
    :param rect: -
    :param Trv2c: -
    :param P2: [4, 4]
    :return:
    """

    # [N, 8, 3]
    box_corners = center_to_corner_box3d(
        box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1)
    # [N, 8, 2]
    box_corners_in_image = project_to_image(
        box_corners, P2)
    # min x and min y[N, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    # max x and max y
    maxxy = np.max(box_corners_in_image, axis=1)
    # concatenated min-x, min-y, max-x, max-y
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above
    to surface coordinates that normal vectors all direct to internal (TODO: external?).

    Args:
        corners (float array, [N, 8, 3]): 3d box corners.
    Returns:
        surfaces (float array, [N, 6, 4, 3]): 6 surfaces in total, each surface consists of 4 points
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


def points_in_rbbox(points, rbbox, lidar=True):
    """find the indices of points inside a 3d bounding box

    :param points: [N_p, 3?]
    :param rbbox:  [N_b, 7]
    :param lidar:  in lidar coordinate or camera coordinate
    :return:  indices of the points inside the bounding box
    """
    if lidar:
        h_axis = 2
        origin = [0.5, 0.5, 0]
    else:
        origin = [0.5, 1.0, 0.5]
        h_axis = 1
    # convert boxes into corners: [N, 8, 3]
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=h_axis)
    # convert corners into surface coordinates: [N, 6, 4, 3]
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    #
    indices = geometry.points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def minmax_to_corner_2d(minmax_box):
    """convert min-max values of 2d bounding box into corner coordinates

    :param minmax_box: [N, 2*ndim] minx, miny, maxx, maxy
    :return: corners: [N, 2**ndim, ndim]
    """
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]  # origin coordinates
    dims = minmax_box[..., ndim:] - center  # dims (lengths)
    # note that origin=0.0
    return center_to_corner_box2d(center, dims, origin=0.0)


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def create_anchors_3d_range(feature_size,
                            anchor_range,
                            sizes=((1.6, 3.9, 1.56),),
                            rotations=(0, np.pi / 2),
                            dtype=np.float32):
    """create anchors with ranges and sizes in [D, H, W] order.

    Args:
        feature_size: [3] number of anchors along each dimension, (reverse order) list [D, H, W] (zyx)
        anchor_range: [6], anchor range min-x,y,z max x,y,z
        sizes: [N, 3] or [1, 3] list of list or array, one of multiple sizes of anchors, xyz
        rotations: rotation choices of anchors
        dtype: data type

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    anchor_range = np.array(anchor_range, dtype)
    # x,y,z coordinates of anchor centers
    z_centers = np.linspace(
        anchor_range[2], anchor_range[5], feature_size[0], dtype=dtype)
    y_centers = np.linspace(
        anchor_range[1], anchor_range[4], feature_size[1], dtype=dtype)
    x_centers = np.linspace(
        anchor_range[0], anchor_range[3], feature_size[2], dtype=dtype)

    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')  # {'ij', 'xy'}, ij mean xyz order
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        # replicate # of anchor sizes times
        rets[i] = np.tile(rets[i][..., np.newaxis, :],
                          tile_shape)   # Nx, Ny, Nz, N_size, Nr
        # for concat,  Nx, Ny, Nz, N_size, Nr, 1
        rets[i] = rets[i][..., np.newaxis]
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    # convert xyz order into zyx order (first 3 dimensions)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


def center_to_minmax_2d_0_5(centers, dims):
    """centers-dims and centers+dims"""
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    TODO: what is this function used for?

    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps
