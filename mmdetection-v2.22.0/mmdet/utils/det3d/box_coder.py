"""bbox encoder/decoder.
For example, the residual coder encodes/decodes the residual between predictions and anchors.
"""
import numpy as np
import torch


class ResidualCoder(object):
    def __init__(self):
        super().__init__()
        self.code_size = 7

    @staticmethod
    def encode_np(boxes, anchors):
        """
        :param boxes: (N, 7) x, y, z, w, l, h, r
        :param anchors: (N, 7)
        :return:
        """
        # need to convert boxes to z-center format
        xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
        xg, yg, zg, wg, lg, hg, rg = np.split(boxes, 7, axis=-1)
        zg = zg + hg / 2
        za = za + ha / 2
        diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha  # 1.6
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
        rt = rg - ra
        return np.concatenate([xt, yt, zt, wt, lt, ht, rt], axis=-1)

    @staticmethod
    def decode_np(box_encodings, anchors):
        """
        :param box_encodings: (N, 7) x, y, z, w, l, h, r
        :param anchors: (N, 7)
        :return:
        """
        # need to convert box_encodings to z-bottom format
        xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
        xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, 7, axis=-1)

        za = za + ha / 2
        diagonal = np.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1)

    @staticmethod
    def encode_torch(boxes, anchors):
        """
        :param boxes: (N, 7) x, y, z, w, l, h, r
        :param anchors: (N, 7)
        :return:
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        :param box_encodings: (N, 7) x, y, z, w, l, h, r
        :param anchors: (N, 7)
        :return:
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

        # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
        za = za + ha / 2
        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


class BinBasedCoder(object):
    def __init__(self, loc_scope, loc_bin_size, num_head_bin,
                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25,
                 get_ry_fine=False, canonical_transform=False):
        super().__init__()
        self.loc_scope = loc_scope
        self.loc_bin_size = loc_bin_size
        self.num_head_bin = num_head_bin
        self.get_xz_fine = get_xz_fine
        self.get_y_by_bin = get_y_by_bin
        self.loc_y_scope = loc_y_scope
        self.loc_y_bin_size = loc_y_bin_size
        self.get_ry_fine = get_ry_fine
        self.canonical_transform = canonical_transform

    def decode_torch(self, pred_reg, roi_box3d, anchor_size):
        """
        decode in LiDAR coordinate
        :param pred_reg: (N, C)
        :param roi_box3d: (N, 7)
        :param anchor_size: ?
        :return:
        """
        anchor_size = anchor_size.to(roi_box3d.get_device())
        per_loc_bin_num = int(self.loc_scope / self.loc_bin_size) * 2
        loc_y_bin_num = int(self.loc_y_scope / self.loc_y_bin_size) * 2

        # recover xz localization
        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r

        x_bin = torch.argmax(pred_reg[:, x_bin_l: x_bin_r], dim=1)
        z_bin = torch.argmax(pred_reg[:, z_bin_l: z_bin_r], dim=1)

        pos_x = x_bin.float() * self.loc_bin_size + self.loc_bin_size / 2 - self.loc_scope
        pos_z = z_bin.float() * self.loc_bin_size + self.loc_bin_size / 2 - self.loc_scope

        if self.get_xz_fine:
            x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
            z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
            start_offset = z_res_r

            x_res_norm = torch.gather(pred_reg[:, x_res_l: x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
            z_res_norm = torch.gather(pred_reg[:, z_res_l: z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
            x_res = x_res_norm * self.loc_bin_size
            z_res = z_res_norm * self.loc_bin_size

            pos_x += x_res
            pos_z += z_res

        # recover y localization
        if self.get_y_by_bin:
            y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
            y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
            start_offset = y_res_r

            y_bin = torch.argmax(pred_reg[:, y_bin_l: y_bin_r], dim=1)
            y_res_norm = torch.gather(pred_reg[:, y_res_l: y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
            y_res = y_res_norm * self.loc_y_bin_size
            pos_y = y_bin.float() * self.loc_y_bin_size + self.loc_y_bin_size / 2 - self.loc_y_scope + y_res
            pos_y = pos_y + roi_box3d[:, 1]
        else:
            y_offset_l, y_offset_r = start_offset, start_offset + 1
            start_offset = y_offset_r

            pos_y = pred_reg[:, y_offset_l]

        # recover ry rotation
        ry_bin_l, ry_bin_r = start_offset, start_offset + self.num_head_bin
        ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + self.num_head_bin

        ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
        ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
        if self.get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / self.num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
        else:
            angle_per_class = (2 * np.pi) / self.num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)

            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
            ry[ry > np.pi] -= 2 * np.pi

        # recover size
        size_res_l, size_res_r = ry_res_r, ry_res_r + 3
        assert size_res_r == pred_reg.shape[1]

        size_res_norm = pred_reg[:, size_res_l: size_res_r]
        wlh = size_res_norm * anchor_size + anchor_size

        # shift to original coords
        roi_center = roi_box3d[:, 0:3]
        # Note: x, z, y, be consistent with get_reg_loss_lidar
        shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_z.view(-1, 1), pos_y.view(-1, 1), wlh, ry.view(-1, 1)), dim=1)
        ret_box3d = shift_ret_box3d
        if self.canonical_transform and roi_box3d.shape[1] == 7:
            roi_ry = roi_box3d[:, 6]
            ret_box3d = rotate_pc_along_z_torch(shift_ret_box3d, (roi_ry + np.pi / 2))
            ret_box3d[:, 6] += roi_ry
        ret_box3d[:, 0:3] += roi_center

        return ret_box3d


def rotate_pc_along_z_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, 0:2].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, 0:2] = torch.matmul(pc_temp, R).squeeze(dim=1)
    return pc


if __name__ == '__main__':
    A = ResidualCoder()
    import pdb
    pdb.set_trace()
