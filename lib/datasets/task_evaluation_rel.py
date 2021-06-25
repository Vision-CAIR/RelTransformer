"""
Coppied from Rowan Zellers, modified by Ji
"""
"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import os
import numpy as np
import logging
from six.moves import cPickle as pickle
import json
import csv
from tqdm import tqdm

from core.config import cfg
from functools import reduce
# from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from utils.boxes import bbox_overlaps, bbox_pair_overlaps, boxes_union
from datasets.voc_eval_rel import voc_eval, prepare_mAP_dets

np.set_printoptions(precision=3)

logger = logging.getLogger(__name__)


topk = 100


def eval_rel_results(all_results, output_dir, do_val):
    
    if cfg.TEST.DATASETS[0].find('vg') >= 0:
        prd_k_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20)
    elif cfg.TEST.DATASETS[0].find('vrd') >= 0:
        prd_k_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 70)
    else:
        prd_k_set = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        
    if cfg.TEST.DATASETS[0].find('vg') >= 0:
        eval_sets = (False,)
    else:
        eval_sets = (False, True)

    for phrdet in eval_sets:
        eval_metric = 'phrdet' if phrdet else 'reldet'  
        print('{}:'.format(eval_metric))

        for prd_k in prd_k_set:
            print('prd_k = {}:'.format(prd_k))

            recalls = {20: 0, 50: 0, 100: 0}
            if do_val:
                all_gt_cnt = 0

            topk_dets = []
            for im_i, res in enumerate(tqdm(all_results)):

                # in oi_all_rel some images have no dets
                if res['prd_scores'] is None:
                    det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
                    det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
                    det_labels_s_top = np.zeros(0, dtype=np.int32)
                    det_labels_p_top = np.zeros(0, dtype=np.int32)
                    det_labels_o_top = np.zeros(0, dtype=np.int32)
                    det_scores_top = np.zeros(0, dtype=np.float32)
                else:
                    det_boxes_sbj = res['sbj_boxes']  # (#num_rel, 4)
                    det_boxes_obj = res['obj_boxes']  # (#num_rel, 4)
                    det_labels_sbj = res['sbj_labels']  # (#num_rel,)
                    det_labels_obj = res['obj_labels']  # (#num_rel,)
                    det_scores_sbj = res['sbj_scores']  # (#num_rel,)
                    det_scores_obj = res['obj_scores']  # (#num_rel,)
                    det_scores_prd = res['prd_scores'][:, 1:]

                    det_labels_prd = np.argsort(-det_scores_prd, axis=1)
                    det_scores_prd = -np.sort(-det_scores_prd, axis=1)

                    det_scores_so = det_scores_sbj * det_scores_obj
                    det_scores_spo = det_scores_so[:, None] * det_scores_prd[:, :prd_k]
                    det_scores_spo = det_scores_prd[:, :prd_k]
                    det_scores_inds = argsort_desc(det_scores_spo)[:topk]
                    det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                    det_boxes_so_top = np.hstack(
                        (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
                    det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]
                    det_labels_spo_top = np.vstack(
                        (det_labels_sbj[det_scores_inds[:, 0]], det_labels_p_top, det_labels_obj[det_scores_inds[:, 0]])).transpose()

#                     cand_inds = np.where(det_scores_top > cfg.TEST.SPO_SCORE_THRESH)[0]
#                     det_boxes_so_top = det_boxes_so_top[cand_inds]
#                     det_labels_spo_top = det_labels_spo_top[cand_inds]
#                     det_scores_top = det_scores_top[cand_inds]

                    det_boxes_s_top = det_boxes_so_top[:, :4]
                    det_boxes_o_top = det_boxes_so_top[:, 4:]
                    det_labels_s_top = det_labels_spo_top[:, 0]
                    det_labels_p_top = det_labels_spo_top[:, 1]
                    det_labels_o_top = det_labels_spo_top[:, 2]

                topk_dets.append(dict(image=res['image'],
                                      det_boxes_s_top=det_boxes_s_top,
                                      det_boxes_o_top=det_boxes_o_top,
                                      det_labels_s_top=det_labels_s_top,
                                      det_labels_p_top=det_labels_p_top,
                                      det_labels_o_top=det_labels_o_top,
                                      det_scores_top=det_scores_top))

                if do_val:
                    gt_boxes_sbj = res['gt_sbj_boxes']  # (#num_gt, 4)
                    gt_boxes_obj = res['gt_obj_boxes']  # (#num_gt, 4)
                    gt_labels_sbj = res['gt_sbj_labels']  # (#num_gt,)
                    gt_labels_obj = res['gt_obj_labels']  # (#num_gt,)
                    gt_labels_prd = res['gt_prd_labels']  # (#num_gt,)
                    gt_boxes_so = np.hstack((gt_boxes_sbj, gt_boxes_obj))
                    gt_labels_spo = np.vstack((gt_labels_sbj, gt_labels_prd, gt_labels_obj)).transpose()
                    # Compute recall. It's most efficient to match once and then do recall after
                    # det_boxes_so_top is (#num_rel, 8)
                    # det_labels_spo_top is (#num_rel, 3)
                    if phrdet:
                        det_boxes_r_top = boxes_union(det_boxes_s_top, det_boxes_o_top)
                        gt_boxes_r = boxes_union(gt_boxes_sbj, gt_boxes_obj)
                        pred_to_gt = _compute_pred_matches(
                            gt_labels_spo, det_labels_spo_top,
                            gt_boxes_r, det_boxes_r_top,
                            phrdet=phrdet)
                    else:
                        pred_to_gt = _compute_pred_matches(
                            gt_labels_spo, det_labels_spo_top,
                            gt_boxes_so, det_boxes_so_top,
                            phrdet=phrdet)
                    all_gt_cnt += gt_labels_spo.shape[0]
                    for k in recalls:
                        if len(pred_to_gt):
                             match = reduce(np.union1d, pred_to_gt[:k])
                        else:
                            match = []
                        recalls[k] += len(match)

                    topk_dets[-1].update(dict(gt_boxes_sbj=gt_boxes_sbj,
                                              gt_boxes_obj=gt_boxes_obj,
                                              gt_labels_sbj=gt_labels_sbj,
                                              gt_labels_obj=gt_labels_obj,
                                              gt_labels_prd=gt_labels_prd))

            if do_val:
                for k in recalls:
                    recalls[k] = float(recalls[k]) / (float(all_gt_cnt) + 1e-12)
                print_stats(recalls)


def print_stats(recalls):
    # print('====================== ' + 'sgdet' + ' ============================')
    for k, v in recalls.items():
        print('R@%i: %f' % (k, v))


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh=0.5, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            rel_iou = bbox_overlaps(gt_box[None, :], boxes)[0]

            inds = rel_iou >= iou_thresh
        else:
            gt_box = gt_box.astype(dtype=np.float32, copy=False)
            boxes = boxes.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
