import torch
import torch.nn.functional as F
import numpy as np

def np_norm_to_pixel(reference_points_cam, image_shape):
    B, num_cam, _, _ = reference_points_cam.shape
    reference_points_cam = (reference_points_cam + 1.) / 2.
    reference_points_cam = reference_points_cam * np.asarray(image_shape)[[1,0]]
    return reference_points_cam

def np_lidar_to_img(lidar2img, points, image_shape, return_front_mask=False, return_inside_img_mask=False):
    """
    :params lidar2img: [B, num_cam, 4, 4]
            points: [B, N, 5]

    :return points_cam: [B, N, 3] 
            mask: [B, num_cam, N]
    """
    reference_points = points[:, :, :3]
    reference_points = np.concatenate((reference_points, np.ones_like(reference_points[..., :1])), -1)

    B, num_query = reference_points.shape[:2]
    num_cam = lidar2img.shape[1]

    reference_points = reference_points.reshape(B, 1, num_query, 4, 1)
    lidar2img = lidar2img.reshape(B, num_cam, 1, 4, 4)

    reference_points_cam = np.einsum('ijkmn,ijknl->ijkml', lidar2img, reference_points)[..., 0]

    eps = 1e-6
    front_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / reference_points_cam[..., 2:3].clip(min=eps)
    reference_points_cam[..., 0] /= image_shape[1]
    reference_points_cam[..., 1] /= image_shape[0]

    reference_points_cam = (reference_points_cam - 0.5) * 2
    reference_points_cam[..., :2] = reference_points_cam[..., :2].clip(min=-1., max=1.)
    inside_img_mask = (reference_points_cam[..., 0:1] > -1.0) \
                    & (reference_points_cam[..., 0:1] < 1.0) \
                    & (reference_points_cam[..., 1:2] > -1.0) \
                    & (reference_points_cam[..., 1:2] < 1.0)

    mask = front_mask & inside_img_mask
    mask = mask.reshape(B, num_cam, num_query)
    # mask = np.nan_to_num(mask)

    ret = [reference_points_cam, mask]
    if return_front_mask:
        front_mask = front_mask.reshape(B, num_cam, num_query)
        # front_mask = np.nan_to_num(front_mask)
        ret.append(front_mask)
    if return_inside_img_mask:
        inside_img_mask = inside_img_mask.reshape(B, num_cam, num_query)
        # inside_img_mask = np.nan_to_num(inside_img_mask)
        ret.append(inside_img_mask)
    return ret

def torch_lidar_to_img(lidar2img, points, image_shape, return_front_mask=False, return_inside_img_mask=False):
    """
    :params lidar2img: [B, num_cam, 4, 4]
            points: [B, N, 5]

    :return points_cam: [B, N, 3] 
            mask: [B, num_cam, N]
    """
    reference_points = points[:, :, :3]
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

    B, num_query = reference_points.shape[:2]
    num_cam = lidar2img.shape[1]
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)

    eps = 1e-6
    front_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    reference_points_cam[..., 0] /= image_shape[1]
    reference_points_cam[..., 1] /= image_shape[0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    reference_points_cam[..., :2] = reference_points_cam[..., :2].clamp(min=-1., max=1.)
    inside_img_mask = (reference_points_cam[..., 0:1] > -1.0) \
                    & (reference_points_cam[..., 0:1] < 1.0) \
                    & (reference_points_cam[..., 1:2] > -1.0) \
                    & (reference_points_cam[..., 1:2] < 1.0)
    
    mask = front_mask & inside_img_mask
    mask = mask.view(B, num_cam, num_query)
    # mask = torch.nan_to_num(mask)

    ret = [reference_points_cam, mask]
    if return_front_mask:
        front_mask = front_mask.view(B, num_cam, num_query)
        # front_mask = np.nan_to_num(front_mask)
        ret.append(front_mask)
    if return_inside_img_mask:
        inside_img_mask = inside_img_mask.view(B, num_cam, num_query)
        # inside_img_mask = np.nan_to_num(inside_img_mask)
        ret.append(inside_img_mask)
    return ret

def torch_mv_grid_sample(reference_points_cam, mask, mv_feats):
    """
    :params 
            points_cam: [B, num_cam, num_query, 3] 
            mask: [B, num_cam, num_query]
            mv_feats: list or [B, num_cam, C, H, W]
    """
    if not isinstance(mv_feats, list):
        mv_feats = [mv_feats]

    assert reference_points_cam.shape[:2] == mv_feats[0].shape[:2]
    assert reference_points_cam.shape[:3] == mask.shape[:3]

    eps = 1e-6
    B, num_cam, num_query, _ = reference_points_cam.shape
    B, num_cam, C, H, W = mv_feats[0].shape

    sampled_feats = []
    for lvl, feat in enumerate(mv_feats):
        feat = feat.view(B*num_cam, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*num_cam, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, num_cam, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, len(mv_feats))
    
    mask = mask.permute(0,2,1)
    sampled_feats = (sampled_feats * mask[:, None, :, :, None]).sum(3)[..., 0] / (mask[:, None, :, :, None].sum(3)[..., 0] + eps)
    return sampled_feats

class MVCalibration(object):
    def __init__(self, V2Cs, Ps):
        self.V2Cs = np.asarray(V2Cs, dtype=np.float32)
        self.P2 = np.asarray(Ps, dtype=np.float32)
        
        self.flipped = False
        self.offsets = [0, 0]

    @property
    def lidar2img(self):
        return self.P2 @ self.V2Cs

    @property
    def cu(self):
        return self.P2[:, 0, 2]

    @property
    def cv(self):
        return self.P2[:, 1, 2]

    @property
    def fu(self):
        return self.P2[:, 0, 0]

    @property
    def fv(self):
        return self.P2[:, 1, 1]

    @property
    def tx(self):
        return self.P2[:, 0, 3] / (-self.fu)

    @property
    def ty(self):
        return self.P2[:, 1, 3] / (-self.fv)

    # @property
    # def txyz(self):
    #     return np.matmul(np.linalg.inv(self.P2[:3, :3]), self.P2[:3, 3:4]).squeeze(-1)

    @property
    def K(self):
        return self.P2[:, :3, :3]

    # @property
    # def K3x4(self):
    #     return np.concatenate([self.P2[:3, :3], np.zeros_like(self.P2[:3, 3:4])], axis=1)

    # @property
    # def K3x4_R(self):
    #     return np.concatenate([self.P3[:3, :3], np.zeros_like(self.P3[:3, 3:4])], axis=1)

    @property
    def inv_K(self):
        return np.linalg.inv(self.K)

    def offset(self, offset_x, offset_y):
        assert np.all((self.P2[:, -1, :] - np.asarray([0., 0., 0., 1.])) < 1e-6)
        assert np.all((self.P2[:, :, -1] - np.asarray([0., 0., 0., 1.])) < 1e-6)
        self.P2[:, 0, 2] -= offset_x
        self.P2[:, 1, 2] -= offset_y
        self.offsets[0] += offset_x
        self.offsets[1] += offset_y

    def fliplr(self, image_width):
        from IPython import embed; embed()

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

    @staticmethod
    def cart_to_hom(pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack(
            (pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    # def rect_to_lidar(self, pts_rect):
    #     """
    #     :param pts_lidar: (N, 3)
    #     :return pts_rect: (N, 3)
    #     """
    #     if self.flipped:
    #         raise NotImplementedError

    #     from IPython import embed; embed()

    #     pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
    #     R0_ext = np.hstack(
    #         (self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    #     R0_ext = np.vstack(
    #         (R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    #     R0_ext[3, 3] = 1
    #     V2C_ext = np.vstack(
    #         (self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
    #     V2C_ext[3, 3] = 1

    #     pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(
    #         np.dot(R0_ext, V2C_ext).T))
    #     return pts_lidar[:, 0:3]

    # @staticmethod
    # def rect_to_lidar_pseudo(pts_rect):
    #     pts_rect_hom = Calibration.cart_to_hom(pts_rect)
    #     T = np.array([[0, 0, 1, 0],
    #                   [-1, 0, 0, 0],
    #                   [0, -1, 0, 0],
    #                   [0, 0, 0, 1]], dtype=np.float32)
    #     pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(T))
    #     return pts_lidar[:, 0:3]

    # def lidar_to_rect(self, pts_lidar):
    #     """
    #     :param pts_lidar: (N, 3)
    #     :return pts_rect: (N, 3)
    #     """
    #     if self.flipped:
    #         raise NotImplementedError
        
    #     from IPython import embed; embed()

    #     pts_lidar_hom = self.cart_to_hom(pts_lidar)
    #     # pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
    #     pts_rect = np.dot(pts_lidar_hom, self.V2C.T)
    #     pts_rect = np.dot(pts_rect, self.R0.T)
    #     # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
    #     return pts_rect

    # @staticmethod
    # def lidar_pseudo_to_rect(pts_lidar):
    #     pts_lidar_hom = Calibration.cart_to_hom(pts_lidar)
    #     T = np.array([[0, 0, 1],
    #                   [-1, 0, 0],
    #                   [0, -1, 0],
    #                   [0, 0, 0]], dtype=np.float32)
    #     pts_rect = np.dot(pts_lidar_hom, T)
    #     return pts_rect

    # def torch_lidar_pseudo_to_rect(self, pts_lidar):
    #     pts_lidar_hom = torch.cat([pts_lidar, torch.ones_like(pts_lidar[..., -1:])], dim=-1)
    #     T = np.array([[0, 0, 1],
    #                   [-1, 0, 0],
    #                   [0, -1, 0],
    #                   [0, 0, 0]], dtype=np.float32)
    #     T = torch.from_numpy(T).cuda()
    #     pts_rect = torch.matmul(pts_lidar_hom, T)
    #     return pts_rect

    # def rect_to_img(self, pts_rect, right=False):
    #     """
    #     :param pts_rect: (N, 3)
    #     :return pts_img: (N, 2)
    #     """
    #     pts_rect_hom = self.cart_to_hom(pts_rect)
    #     P = (self.P2 if not right else self.P3)
    #     pts_2d_hom = np.dot(pts_rect_hom, P.T)
    #     pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
    #     pts_rect_depth = pts_2d_hom[:, 2] - \
    #         P.T[3, 2]  # depth in rect camera coord
    #     return pts_img, pts_rect_depth
    
    # def torch_rect_to_img(self, pts_rect):
    #     """
    #     :param pts_rect: (N, 3)
    #     :return pts_img: (N, 2)
    #     """
    #     pts_rect_hom = torch.cat([pts_rect, torch.ones_like(pts_rect[..., -1:])], dim=-1)
    #     pts_2d_hom = torch.matmul(pts_rect_hom, torch.from_numpy(self.P2.T).cuda())
    #     pts_img = pts_2d_hom[..., 0:2] / pts_rect_hom[..., 2:3]
    #     return pts_img

    # def lidar_to_img(self, pts_lidar):
    #     """
    #     :param pts_lidar: (N, 3)
    #     :return pts_img: (N, 2)
    #     """
    #     if self.flipped:
    #         raise NotImplementedError
    #     pts_rect = self.lidar_to_rect(pts_lidar)
    #     pts_img, pts_depth = self.rect_to_img(pts_rect)
    #     return pts_img, pts_depth

    # def img_to_rect(self, u, v, depth_rect):
    #     """
    #     :param u: (N)
    #     :param v: (N)
    #     :param depth_rect: (N)
    #     :return:
    #     """
    #     x = ((u - self.cu) * depth_rect) / self.fu + self.tx
    #     y = ((v - self.cv) * depth_rect) / self.fv + self.ty
    #     pts_rect = np.concatenate(
    #         (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
    #     return pts_rect

    # def corners3d_to_img_boxes(self, corners3d):
    #     """
    #     :param corners3d: (N, 8, 3) corners in rect coordinate
    #     :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
    #     :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
    #     """
    #     sample_num = corners3d.shape[0]
    #     corners3d_hom = np.concatenate(
    #         (corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

    #     img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

    #     x, y = img_pts[:, :, 0] / img_pts[:, :,
    #                                       2], img_pts[:, :, 1] / img_pts[:, :, 2]
    #     x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
    #     x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

    #     boxes = np.concatenate(
    #         (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
    #     boxes_corner = np.concatenate(
    #         (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

    #     return boxes, boxes_corner

