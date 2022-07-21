import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

import cv2

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

    if 'depth_error_all_local_median' in ret_dict:
        if not isinstance(ret_dict['depth_error_all_local_median'], list):
            for k, v in ret_dict.items():
                if k.startswith('depth_error_'):
                    ret_dict[k] = [ret_dict[k]]
        metric['num'] += len(ret_dict['depth_error_all_local_median'])

        # depth evaluation for stereo detection
        for b in range(len(ret_dict['depth_error_all_local_median'])):
            for k, v in ret_dict.items():
                if k.startswith('depth_error_'):
                    if k.endswith('perbox'):
                        if k not in metric:
                            metric[k] = []
                        metric[k].extend(v[b])
                    else:
                        if isinstance(v[b], torch.Tensor):
                            val = v[b].item()
                        else:
                            val = v[b]
                        metric[k] = metric.get(k, 0.) + val
                if k in ['depth_error_fg_median', 'depth_error_median']:
                    disp_dict[k] = '%.3f' % (metric[k] / metric['num'])

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    final_2d_output_dir = result_dir / 'final_result' / 'data2d'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        final_2d_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'num': 0,
        'gt_num': 0,
        # 'depth_error_mean': 0.,
        # 'depth_error_median': 0.,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    det_annos_2d = []
    iou_results = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        if 'gt_boxes' in batch_dict and 'iou_results' in pred_dicts[0]:
            iou_results.extend([x['iou_results'] for x in pred_dicts])
        annos_2d = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_2d_output_dir if save_to_file else None,
            mode_2d=True
        ) if 'pred_scores_2d' in pred_dicts[0] else None

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        ) if 'pred_scores' in pred_dicts[0] else None

        if False:
            from IPython import embed; embed()
            from pcdet.utils import box_utils
            from pcdet.ops.iou3d_nms import iou3d_nms_utils
            from pcdet.utils.open3d_utils import save_point_cloud, save_box_corners
            pred_box_corners = box_utils.boxes_to_corners_3d(pred_dicts[0]['pred_boxes'].cpu().numpy())
            gt_box_corners = box_utils.boxes_to_corners_3d(batch_dict['gt_boxes'][0, :, :7].cpu().numpy())
            points = batch_dict['points'][:, 1:].cpu().numpy()
            save_point_cloud(points, 'temp/pc.ply')
            save_box_corners(gt_box_corners, 'temp/gt_box.ply')
            save_box_corners(pred_box_corners, 'temp/pred_box.ply')

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            # img = (img.astype(np.float32) / 255 - mean) / std
            img = batch_dict['left_img'][0].permute(1, 2, 0).cpu().numpy()
            img = (((img * std) + mean) * 255).astype(np.uint8)
            cv2.imwrite('temp/left_img.jpg', img)
            
        if annos_2d is not None:
            det_annos_2d += annos_2d
        if annos is not None:
            det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        iou_results = common_utils.merge_results_dist(iou_results, len(dataset), tmpdir=result_dir / 'tmpdir')
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        det_annos_2d = common_utils.merge_results_dist(det_annos_2d, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall
    
    for k in metric:
        if k.startswith('depth_error_'):
            if not k.endswith('perbox'):
                metric[k] /= metric['num']
                logger.info('%s: %f' % (k, metric[k]))
                ret_dict['depth_error/%s' % (k)] = metric[k]
            else:
                if len(metric[k]) > 0:
                    for kk in metric[k][0]:
                        if kk.startswith("err_"):
                            values = [item[kk] for item in metric[k]]
                            mean_value = np.mean(values)
                            logger.info('%s: %f' % (k + "_" + kk, mean_value))
                            ret_dict['%s' % (k + "_" + kk)] = mean_value

                # copy iou into metric[k]
                if not iou_results:
                    continue

                for x in metric[k]:
                    x['iou'] = iou_results[x['image_idx']][x['idx']]

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)
    with open(result_dir / 'metric_result.pkl', 'wb') as f:
        pickle.dump(metric, f)

    if det_annos and 'gt_boxes' in batch_dict:
        logger.info('---- 3d box evaluation ---- ')
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric='3d',
            output_path=final_output_dir
        )
        logger.info(result_str)
        ret_dict.update(result_dict)
    if det_annos_2d and 'gt_boxes_2d' in batch_dict:
        logger.info('---- 2d box evaluation ---- ')
        result_str, _ = dataset.evaluation(
            det_annos_2d, class_names,
            eval_metric='2d',
            output_path=final_2d_output_dir
        )
        logger.info(result_str)
    else:
        logger.info(f"no 2d eval: {'gt_boxes_2d' in batch_dict} / {det_annos_2d}")

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
