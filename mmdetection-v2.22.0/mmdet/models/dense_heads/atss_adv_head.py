import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmcv.ops import batched_nms
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .atss_head import ATSSHead

def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

@HEADS.register_module()
class ATSSAdvHead(ATSSHead):
    """
    Expand ATSS head with some extra functions
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 reg_decoded_bbox=False,
                 reg_class_agnostic=True,
                 num_extra_reg_channel=0,
                 seperate_extra_reg_branch=False,
                 **kwargs):
        assert not seperate_extra_reg_branch
        self.reg_class_agnostic = reg_class_agnostic
        assert num_extra_reg_channel == 0, 'extra reg not used'
        self.num_reg_channel = num_extra_reg_channel + 4
        super(ATSSAdvHead, self).__init__(num_classes,
                                          in_channels,
                                          stacked_convs=stacked_convs,
                                          conv_cfg=conv_cfg,
                                          norm_cfg=norm_cfg,
                                          reg_decoded_bbox=reg_decoded_bbox,
                                          **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size)
        if self.reg_class_agnostic:
            self.atss_reg = nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.num_reg_channel,
                self.pred_kernel_size,
                padding=pred_pad_size)
        else:
            self.atss_reg = nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.num_reg_channel * self.num_classes,
                self.pred_kernel_size,
                padding=pred_pad_size)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        if self.reg_class_agnostic:
            bbox_pred = bbox_pred.permute(
                0, 2, 3, 1).reshape(-1, self.num_reg_channel)
        else:
            bbox_pred = bbox_pred.permute(
                0, 2, 3, 1).reshape(-1, self.num_reg_channel, self.num_classes)
            # add a blank bg class, --> [N, 4, N_cls+1]
            bbox_pred = torch.cat(
                [bbox_pred, torch.zeros_like(bbox_pred[:, :, :1])], dim=-1)
            bbox_pred = torch.gather(
                bbox_pred, dim=2, index=labels.reshape(-1, 1, 1).repeat(1, self.num_reg_channel, 1)).squeeze(-1)

        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, self.num_reg_channel)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)
            centerness_targets = self.centerness_target(
                pos_anchors, pos_decode_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            if self.reg_class_agnostic:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            else:
                max_cls_indices = scores.argmax(1)
                bbox_pred = bbox_pred.permute(
                    1, 2, 0).reshape(-1, self.num_reg_channel, self.num_classes)
                # add a blank bg class, --> [N, 4, N_cls+1]
                bbox_pred = torch.cat(
                    [bbox_pred, torch.zeros_like(bbox_pred[:, :, :1])], dim=-1)
                bbox_pred = torch.gather(
                    bbox_pred, dim=2, index=max_cls_indices.reshape(-1, 1, 1).repeat(1, self.num_reg_channel, 1)).squeeze(-1)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels
