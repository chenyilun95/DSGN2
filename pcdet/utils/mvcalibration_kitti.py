import torch
import numpy as np

from .calibration_kitti import *

class MVCalibration(Calibration):
    #----------------- Multi-frame Projections ----------------
    def init_poses(self, poses):
        hom_poses = np.concatenate([poses, np.zeros((len(poses), 1, 4))], axis=1)
        hom_poses[:, 3, 3] = 1.
        self.hom_poses = hom_poses.copy()
    
    def pose_transform(self, start_frame, end_frame, pts_rect=None):
        if start_frame == end_frame:
            t = np.diag(np.ones(4))
        else:
            t = np.linalg.inv(self.hom_poses[end_frame]).dot(self.hom_poses[start_frame])
        if pts_rect is None:
            return t
        else:
            return self.cart_to_hom(pts_rect).dot(t.T)[:, :3]

    def cu(self, fid): # [0, 1, 2, 3] # last is current frame
        assert fid == -1
        return self.P2[0, 2]

    def cv(self, fid):
        assert fid == -1
        return self.P2[1, 2]

    def fu(self, fid):
        assert fid == -1
        return self.P2[0, 0]

    def fv(self, fid):
        assert fid == -1
        return self.P2[1, 1]

    def tx(self, fid):
        assert fid == -1
        return self.P2[0, 3] / (-self.fu(fid))

    def ty(self, fid):
        assert fid == -1
        return self.P2[1, 3] / (-self.fv(fid))

    def txyz(self, fid):
        assert fid == -1
        return np.matmul(np.linalg.inv(self.P2[:3, :3]), self.P2[:3, 3:4]).squeeze(-1)

    def K(self, fid):
        assert fid == -1
        return self.P2[:3, :3]

    def K3x4(self, fid):
        assert fid == -1
        return np.concatenate([self.P2[:3, :3], np.zeros_like(self.P2[:3, 3:4])], axis=1)

    def K3x4_R(self, fid):
        assert fid == -1
        return np.concatenate([self.P3[:3, :3], np.zeros_like(self.P3[:3, 3:4])], axis=1)

    def inv_K(self, fid):
        return np.linalg.inv(self.K(fid))

    def global_scale(self, scale_factor):
        raise NotImplementedError

    def scale(self, scale_factor):
        raise NotImplementedError

    def offset(self, offset_x, offset_y):
        K = self.P2[:3, :3].copy()
        inv_K = self.inv_K(-1)
        T2 = np.matmul(inv_K, self.P2)
        T3 = np.matmul(inv_K, self.P3)
        K[0, 2] -= offset_x
        K[1, 2] -= offset_y
        self.P2 = np.matmul(K, T2)
        self.P3 = np.matmul(K, T3)
        self.offsets[0] += offset_x
        self.offsets[1] += offset_y

    def fliplr(self, image_width):
        # mirror using y-z plane of cam 0
        assert not self.flipped

        K = self.P2[:3, :3].copy()
        inv_K = np.linalg.inv(K)
        T2 = np.matmul(inv_K, self.P2)
        T3 = np.matmul(inv_K, self.P3)
        T2[0, 3] *= -1
        T3[0, 3] *= -1

        K[0, 2] = image_width - 1 - K[0, 2]
        self.P3 = np.matmul(K, T2)
        self.P2 = np.matmul(K, T3)

        # delete useless parameters to avoid bugs
        del self.R0, self.V2C

        self.flipped = not self.flipped

    @property
    def fu_mul_baseline(self):
        return np.abs(self.P2[0, 3] - self.P3[0, 3])

    def rect_to_img(self, pts_rect, fid=-1, right=False):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect).dot(self.pose_transform(-1, fid).T)

        P = (self.P2 if not right else self.P3)
        pts_2d_hom = np.dot(pts_rect_hom, P.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - \
            P.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def torch_rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        # pts_rect_hom = torch.cat([pts_rect, torch.ones_like(pts_rect[..., -1:])], dim=-1)
        # pts_2d_hom = torch.matmul(pts_rect_hom, torch.from_numpy(self.P2.T).cuda())
        # pts_img = pts_2d_hom[..., 0:2] / pts_rect_hom[..., 2:3]
        # return pts_img
        raise NotImplementedError

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        # if self.flipped:
        #     raise NotImplementedError
        # pts_rect = self.lidar_to_rect(pts_lidar)
        # pts_img, pts_depth = self.rect_to_img(pts_rect)
        # return pts_img, pts_depth
        raise NotImplementedError

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        # x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        # y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        # pts_rect = np.concatenate(
        #     (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        # return pts_rect
        raise NotImplementedError

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        # sample_num = corners3d.shape[0]
        # corners3d_hom = np.concatenate(
        #     (corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        # img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        # x, y = img_pts[:, :, 0] / img_pts[:, :,
        #                                   2], img_pts[:, :, 1] / img_pts[:, :, 2]
        # x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        # x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        # boxes = np.concatenate(
        #     (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        # boxes_corner = np.concatenate(
        #     (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        # return boxes, boxes_corner
        raise NotImplementedError



