"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
from six.moves import cPickle as pickle

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine_mem import get_features_for_centroids
import utils.logging

from datasets import task_evaluation_rel as task_evaluation

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Generate centroids to be used for memory module')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument(
        '--do_val', dest='do_val', help='do evaluation', action='store_true')
    parser.add_argument(
        '--use_gt_boxes', dest='use_gt_boxes', help='use gt boxes for sgcls/prdcls', action='store_true')
    parser.add_argument(
        '--use_gt_labels', dest='use_gt_labels', help='use gt boxes for sgcls/prdcls', action='store_true')    
    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)


    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "vrd":
        cfg.TEST.DATASETS = ('vrd_train',)
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70  # exclude background
    elif args.dataset == "vg":
        cfg.TEST.DATASETS = ('vg_train',)
        cfg.MODEL.NUM_CLASSES = 151
        cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background
    elif args.dataset == "vg80k":
        cfg.TEST.DATASETS = ('vg80k_train',)
        cfg.MODEL.NUM_CLASSES = 53305 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 29086  # excludes background
    elif args.dataset == "gvqa":
        cfg.TEST.DATASETS = ('gvqa_train',)
        cfg.MODEL.NUM_CLASSES = 1704 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 310  # exclude background
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()
    
    if not cfg.MODEL.RUN_BASELINE:
        assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
            'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    if args.use_gt_boxes:
        if args.use_gt_labels:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_prdcls.pkl')
        else:
            det_file = os.path.join(args.output_dir, 'rel_detections_gt_boxes_sgcls.pkl')
    else:
        det_file = os.path.join(args.output_dir, 'rel_detections.pkl')
    all_results = get_features_for_centroids(args)
