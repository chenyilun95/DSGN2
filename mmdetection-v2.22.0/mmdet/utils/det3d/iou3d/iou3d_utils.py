import torch
import iou3d_cuda
import utils.kitti_utils as kitti_utils


def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1)
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1)
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1.0)

    return iou3d


def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


if __name__ == '__main__':
    import numpy as np
    data_dir = '/home/shaoshuai/workspace/cvpr2019/PointRCNN/output/rpn/ex9_fl_aug4_lrex1/eval/epoch_120/smalltrain/'
    rois1_list = kitti_utils.get_objects_from_label(data_dir + 'detections/data/000003.txt')
    rois2_list = kitti_utils.get_objects_from_label(data_dir + 'detections/data/000007.txt')

    num1 = 280
    rois1 = kitti_utils.objs_to_boxes3d(rois1_list[0:num1])
    rois2 = kitti_utils.objs_to_boxes3d(rois1_list[0:200])

    # iou3d_b = kitti_utils.get_iou3d(kitti_utils.boxes3d_to_corners3d(rois1),
    #                                 kitti_utils.boxes3d_to_corners3d(rois2))

    rois1 = torch.from_numpy(rois1).cuda()
    rois2 = torch.from_numpy(rois2).cuda()


    scores1 = kitti_utils.objs_to_scores(rois1_list[0:num1])
    scores1 = torch.from_numpy(scores1).cuda()

    import time
    st = time.time()
    num = 10000
    rois1 = torch.randn(num, 7).cuda()
    scores1 = torch.randn(num).cuda()
    for i in range(500):
        boxes1 = kitti_utils.boxes3d_to_bev_torch(rois1)
        import pdb
        pdb.set_trace()
        keep = nms_gpu(boxes1, scores1, 0.8)

    print('Time: %f' % (time.time() - st))

    ans = boxes1[keep]

    print('ans: %d' % ans.__len__())
    # iou3d_c = kitti_utils.get_iou3d_gpu(rois1, rois2)
    # import pdb
    # pdb.set_trace()

    # for i in range(1000):
    #     print(i)
    #     iou3d_a = boxes_iou3d_gpu(rois1, rois2)
    #
    # print('Done', iou3d_a)
    # print(iou3d_a)
    # x = 1
