import numpy as np
import os
from PIL import Image

def points_to_depth_map(pts_rect, img_shape, calib):
    depth_gt_img = np.zeros(img_shape, dtype=np.float32)
    pts_img, pts_depth = calib.rect_to_img(pts_rect[:, :3])
    iy, ix = np.round(pts_img[:, 1]).astype(np.int64), np.round(pts_img[:, 0]).astype(np.int64)
    mask = (iy >= 0) & (ix >= 0) & (iy < depth_gt_img.shape[0]) & (ix < depth_gt_img.shape[1])
    iy, ix = iy[mask], ix[mask]
    depth_gt_img[iy, ix] = pts_depth[mask]
    return depth_gt_img

def points_to_depth_deviation_map(pts_rect, img_shape, calib):
    depth_gt_img = np.zeros(img_shape, dtype=np.float32)
    pts_img, pts_depth = calib.rect_to_img(pts_rect[:, :3])
    iy, ix = np.round(pts_img[:, 1]).astype(np.int64), np.round(pts_img[:, 0]).astype(np.int64)
    
    # deviation to float
    deviation_gt_img = np.zeros((*img_shape, 2), dtype=np.float32)
    dev_iy, dev_ix = pts_img[:, 1] - iy, pts_img[:, 0] - ix

    mask = (iy >= 0) & (ix >= 0) & (iy < depth_gt_img.shape[0]) & (ix < depth_gt_img.shape[1])
    iy, ix = iy[mask], ix[mask]
    dev_iy, dev_ix = dev_iy[mask], dev_ix[mask]

    depth_gt_img[iy, ix] = pts_depth[mask]
    deviation_gt_img[iy, ix] = np.stack([dev_ix, dev_iy], axis=-1)
    return depth_gt_img, deviation_gt_img

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth
