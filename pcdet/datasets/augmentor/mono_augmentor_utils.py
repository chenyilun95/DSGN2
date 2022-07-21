# Modified from CaDDN. https://github.com/TRAILab/CaDDN
# Monocular augmentation utility functions.

import copy
import numpy as np

def flip_horizontal_mono(calib, image, gt_boxes, points, completion_points=None):
    W = image.shape[1]
    aug_image = np.fliplr(image)

    pts_rect = calib.lidar_pseudo_to_rect(points)
    points_uv, points_depth = calib.rect_to_img(pts_rect)
    points_uv[:, 0] = W - points_uv[:, 0]
    pts_rect = calib.img_to_rect(u=points_uv[:, 0], v=points_uv[:, 1], depth_rect=points_depth)
    pts_lidar = calib.rect_to_lidar_pseudo(pts_rect)
    aug_points = pts_lidar

    if completion_points:
        pts_rect = calib.lidar_pseudo_to_rect(completion_points)
        points_uv, points_depth = calib.rect_to_img(pts_rect)
        points_uv[:, 0] = W - points_uv[:, 0]
        pts_rect = calib.img_to_rect(u=points_uv[:, 0], v=points_uv[:, 1], depth_rect=points_depth)
        pts_lidar = calib.rect_to_lidar_pseudo(pts_rect)
        aug_completion_points = pts_lidar
    else:
        aug_completion_points = None
    
    from pcdet.utils.box_utils import corners_to_boxes_camera, boxes_to_corners_3d, boxes3d_kitti_camera_to_lidar
    aug_gt_boxes2 = copy.copy(gt_boxes)
    lidar_corners = boxes_to_corners_3d(aug_gt_boxes2)
    rect_corners = calib.lidar_pseudo_to_rect(lidar_corners.reshape(-1, 3)).reshape(-1, 3)
    img_pts, img_depth = calib.rect_to_img(rect_corners)
    img_pts[:, 0] = W - img_pts[:, 0]
    pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth).reshape(-1, 8, 3)
    aug_gt_boxes2 = [] 
    for c in pts_rect:
        aug_gt_boxes2.append(corners_to_boxes_camera(c))
    aug_gt_boxes2 = np.asarray(aug_gt_boxes2)
    aug_gt_boxes2 = boxes3d_kitti_camera_to_lidar(aug_gt_boxes2, calib, pseudo_lidar=True)

    # aug_gt_boxes = copy.copy(gt_boxes)
    # locations = aug_gt_boxes[:, :3]
    # img_pts, img_depth = calib.rect_to_img(calib.lidar_pseudo_to_rect(locations))
    # img_pts[:, 0] = W - img_pts[:, 0]
    # pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
    # pts_lidar = calib.rect_to_lidar_pseudo(pts_rect)
    # aug_gt_boxes[:, :3] = pts_lidar
    # aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    return aug_image, aug_gt_boxes2, aug_points, aug_completion_points
