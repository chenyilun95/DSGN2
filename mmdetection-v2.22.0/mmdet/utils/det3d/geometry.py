"""some utility functions for geometry calculation."""
import numpy as np
import numba


def surface_equ_3d(polygon_surfaces):
    """ compute the normal vector and d (3d surface equation parameters) of a set of polygons

    Given a series of polygon points x, y, z, compute the corresponding plane parameters a, b, c and d, which
    satisfies ax+by+cz+d=0, and the normal vector (a, b, c) should point to the inner part of the object.
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]

    :param polygon_surfaces: array [N_poly, N_surfaces, N_points_of_a_surface, 3]
    :return: normal vector and d (surface equation parameters): array [N_poly, N_num_surface, 3/1]
    """
    # compute the edge vector v0->v1 and v1->v2, [num_polygon, num_surfaces, 2, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3], [num_polygon, num_surfaces, 3], the normal vec points to the inner space of the object
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :]), pick a random point to compute the offset d
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces):
    """check points is in 3d convex polygons.

    :param points: [N_points, 3]
    :param polygon_surfaces: [N_poly, max_num_surfaces, max_num_points_of_surface, 3]
    :param normal_vec: [N_poly, max_num_surfaces, 3]
    :param d: [N_poly, max_num_surfaces]
    :param num_surfaces:  [N_poly], may not be used, just set to a large number like 99999
    :return: bool array: [N_point, N_poly]
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_poly = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_poly), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):  # for each point
        for j in range(num_poly):  # for each polyhedron
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3] array.
            all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]  # actually num of polyhedron
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])  # normal vec can be computed with only 3 points
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d, num_surfaces)


@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """check points is in 2d convex polygons. True when point in polygon
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = polygon - polygon[:, [num_points_of_polygon - 1] +
                                 list(range(num_points_of_polygon - 1)), :]
    else:
        vec1 = polygon[:, [num_points_of_polygon - 1] +
                       list(range(num_points_of_polygon - 1)), :] - polygon
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret
