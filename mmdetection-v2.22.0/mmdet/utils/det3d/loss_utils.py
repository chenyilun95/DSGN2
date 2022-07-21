"""Useful loss functions."""
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.box_torch_ops as box_torch_ops


def get_bin_based_reg_loss_lidar(pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25,
                                 get_ry_fine=False):
    """
    :param pred_reg: (N, C)
    :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3)
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    reg_loss_dict = {}
    loc_loss = 0

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 2], reg_label[:, 1]
    x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    x_bin_label = (x_shift / loc_bin_size).floor().long()
    z_bin_label = (z_shift / loc_bin_size).floor().long()

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    loss_x_bin = F.cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
    loss_z_bin = F.cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
    reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
    loc_loss += loss_x_bin + loss_z_bin

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
        z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

        loss_x_res = F.smooth_l1_loss((pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
        loss_z_res = F.smooth_l1_loss((pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.item()
        reg_loss_dict['loss_z_res'] = loss_z_res.item()
        loc_loss += loss_x_res + loss_z_res

    # y localization loss
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = torch.clamp(y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 1e-3)
        y_bin_label = (y_shift / loc_y_bin_size).floor().long()
        y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y_bin_onehot = torch.cuda.FloatTensor(y_bin_label.size(0), loc_y_bin_num).zero_()
        y_bin_onehot.scatter_(1, y_bin_label.view(-1, 1).long(), 1)

        loss_y_bin = F.cross_entropy(pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
        loss_y_res = F.smooth_l1_loss((pred_reg[:, y_res_l: y_res_r] * y_bin_onehot).sum(dim=1), y_res_norm_label)

        reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
        reg_loss_dict['loss_y_res'] = loss_y_res.item()

        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = F.smooth_l1_loss(pred_reg[:, y_offset_l: y_offset_r].sum(dim=1), y_offset_label)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6]

    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

        shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min=1e-3, max=np.pi * 0.5 - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
    ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
    loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1], size_res_r)

    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)

    # Total regression loss
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss

    return loc_loss, angle_loss, size_loss, reg_loss_dict


def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return losses


def get_corner_loss_lidar(pred_bbox3d, gt_bbox3d):
    """
    :param pred_bbox3d: (N, 7)
    :param gt_bbox3d: (N, 7)
    :return: corner_loss: (N)
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_torch_ops.center_to_corner_box3d(
        pred_bbox3d[:, 0:3], pred_bbox3d[:, 3:6], pred_bbox3d[:, 6], [0.5, 0.5, 0], axis=2)
    gt_box_corners = box_torch_ops.center_to_corner_box3d(
        gt_bbox3d[:, 0:3], gt_bbox3d[:, 3:6], gt_bbox3d[:, 6], [0.5, 0.5, 0], axis=2)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_torch_ops.center_to_corner_box3d(
        gt_bbox3d_flip[:, 0:3], gt_bbox3d_flip[:, 3:6], gt_bbox3d_flip[:, 6], [0.5, 0.5, 0], axis=2)

    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))  # (N, 8)
    corner_loss = huber_loss(corner_dist, delta=1.0)  # (N, 8)

    return corner_loss.mean(dim=1)


class Loss(object):
    """Abstract base class for loss functions."""
    __metaclass__ = ABCMeta

    def __call__(self,
                 prediction_tensor,
                 target_tensor,
                 ignore_nan_targets=False,
                 scope=None,
                 **params):
        """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor),
                                        prediction_tensor,
                                        target_tensor)
        return self._compute_loss(prediction_tensor, target_tensor, **params)

    @abstractmethod
    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
        pass


class WeightedSmoothL1LocalizationLoss(Loss):
    """Smooth L1 localization loss function.
    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=np.float32)
            self._code_weights = torch.from_numpy(self._code_weights).cuda()
        else:
            self._code_weights = None
        self._codewise = codewise

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.
        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            code_weights = self._code_weights.type_as(prediction_tensor)
            diff = code_weights.view(1, 1, -1) * diff
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
               + (abs_diff - 0.5 / (self._sigma ** 2)) * (1. - abs_diff_lt_1)
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)  # * weights
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        return anchorwise_smooth_l1norm


class WeightedSoftmaxClassificationLoss(Loss):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """
        self._logit_scale = logit_scale

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(
            prediction_tensor, self._logit_scale)
        per_row_cross_ent = (_softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes)))
        return per_row_cross_ent.view(weights.shape) * weights


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=np.float32):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.

    Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense


class SigmoidFocalClassificationLoss(Loss):
    """Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
          gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
          alpha: optional alpha weighting factor to balance positives vs negatives.
          all_zero_negative: bool. if True, will treat all zero as background.
                else, will treat first label as background. only affect alpha.
        """
        self._alpha = alpha
        self._gamma = gamma

    def _compute_loss(self,
                      prediction_tensor,
                      target_tensor,
                      weights,
                      class_indices=None):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]
          class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        if weights.shape.__len__() == 2:
            weights = weights.unsqueeze(2)
        # if class_indices is not None:
        #     weights *= indices_to_dense_vector(class_indices, prediction_tensor.shape[2]
        #                                        ).view(1, 1, -1).type_as(prediction_tensor)
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha +
                                   (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


class SigmoidFocalClassificationLoss2(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss


def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param) # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss
