""" Training script for steps_with_decay policy"""

import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.blob as blob_utils
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from roi_data.loader_rel import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats_rel import TrainingStats
from evaluation.generate_detections_csv import generate_csv_file_from_det_obj
from evaluation.frequency_based_analysis_of_methods import get_metrics_from_csv
import json
import pprint

#from datasets.json_dataset_rel import JsonDataset
# Set up logging and load config options
logger = setup_logging(__name__)
logging.getLogger('roi_data.loader').setLevel(logging.INFO)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.manual_seed(cfg.RNG_SEED)
np.random.seed(cfg.RNG_SEED)

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=10, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)
    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)

    # Epoch
    parser.add_argument(
        '--start_step',
        help='Starting step count for training epoch. 0-indexed.',
        default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')

    parser.add_argument(
        '--load_ckpt_dir', help='checkpoint dir to load')

    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    parser.add_argument(
        '--seed', dest='seed',
        help='Value of seed here will overwrite seed in cfg file',
        type=int)
    return parser.parse_args()


def save_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'model_step{}.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def save_best_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'best.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def save_eval_ckpt(output_dir, args, step, train_size, model, optimizer):
    """Save checkpoint"""
    if args.no_save:
        return
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_name = os.path.join(ckpt_dir, 'eval.pth'.format(step))
    if isinstance(model, mynn.DataParallel):
        model = model.module
    model_state_dict = model.state_dict()
    torch.save({
        'step': step,
        'train_size': train_size,
        'batch_size': args.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}, save_name)
    logger.info('save model: %s', save_name)


def get_checkpoint_resume_file(checkpoint_dir):
    all_files = os.listdir(checkpoint_dir)
    all_iters = []
    for f in all_files:
        if 'model_step' in f:
            iter_num = int(f.replace('.pth', '').replace('model_step', ''))
            all_iters.append(iter_num)
    if len(all_iters) > 0:
        all_iters.sort(reverse=True)
        last_iter = int(all_iters[0])
        filepath = os.path.join(
            checkpoint_dir, 'model_step{}.pth'.format(last_iter)
        )
        return filepath
    else:
        return None


def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    cfg.DATASET = args.dataset
    if args.dataset == "vg80k":
        cfg.TRAIN.DATASETS = ('vg80k_train',)
        cfg.TEST.DATASETS = ('vg80k_val',)
        cfg.MODEL.NUM_CLASSES = 53305 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 29086  # excludes background
    elif args.dataset == "vg8k":
        cfg.TRAIN.DATASETS = ('vg8k_train',)
        cfg.TEST.DATASETS = ('vg8k_val',)
        cfg.MODEL.NUM_CLASSES = 5331 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 2000  # excludes background
    elif args.dataset == "vrd":
        cfg.TRAIN.DATASETS = ('vrd_train',)
        cfg.TEST.DATASETS = ('vrd_val',)
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70  # exclude background
    elif args.dataset == "vg":
        cfg.TRAIN.DATASETS = ('vg_train',)
        cfg.TEST.DATASETS = ('vg_val',)
        cfg.MODEL.NUM_CLASSES = 151
        cfg.MODEL.NUM_PRD_CLASSES = 50  # exclude background
    elif args.dataset == "gvqa20k":
        cfg.TRAIN.DATASETS = ('gvqa20k_train',)
        cfg.TEST.DATASETS = ('gvqa20k_val',)
        cfg.MODEL.NUM_CLASSES = 1704 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 310  # exclude background
    elif args.dataset == "gvqa10k":
        cfg.TRAIN.DATASETS = ('gvqa10k_train',)
        cfg.TEST.DATASETS = ('gvqa10k_val',)
        cfg.MODEL.NUM_CLASSES = 1704 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 310  # exclude background
    elif args.dataset == "gvqa":
        cfg.TRAIN.DATASETS = ('gvqa_train',)
        cfg.TEST.DATASETS = ('gvqa_val',)
        cfg.MODEL.NUM_CLASSES = 1704 # includes background
        cfg.MODEL.NUM_PRD_CLASSES = 310  # exclude background

    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))

    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.seed:
        cfg.RNG_SEED = args.seed

    # Some imports need to be done after loading the config to avoid using default values
    from datasets.roidb_rel import combined_roidb_for_training
    from modeling.model_builder_rel import Generalized_RCNN
    from core.test_engine_rel import run_eval_inference, run_inference
    from core.test_engine_rel import get_inference_dataset, get_roidb_and_dataset

    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))

    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    effective_batch_size = args.iter_size * args.batch_size
    print('effective_batch_size = batch_size * iter_size = %d * %d' % (args.batch_size, args.iter_size))

    print('Adaptive config changes:')
    print('    effective_batch_size: %d --> %d' % (original_batch_size, effective_batch_size))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

    ### Adjust learning based on batch size change linearly
    # For iter_size > 1, gradients are `accumulated`, so lr is scaled based
    # on batch_size instead of effective_batch_size
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch_size change:\n'
          '    BASE_LR: {} --> {}'.format(old_base_lr, cfg.SOLVER.BASE_LR))

    ### Adjust solver steps
    step_scale = original_batch_size / effective_batch_size
    old_solver_steps = cfg.SOLVER.STEPS
    old_max_iter = cfg.SOLVER.MAX_ITER
    cfg.SOLVER.STEPS = list(map(lambda x: int(x * step_scale + 0.5), cfg.SOLVER.STEPS))
    cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * step_scale + 0.5)
    print('Adjust SOLVER.STEPS and SOLVER.MAX_ITER linearly based on effective_batch_size change:\n'
          '    SOLVER.STEPS: {} --> {}\n'
          '    SOLVER.MAX_ITER: {} --> {}'.format(old_solver_steps, cfg.SOLVER.STEPS,
                                                  old_max_iter, cfg.SOLVER.MAX_ITER))

    if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
        cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
        print('Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
              '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(cfg.FPN.RPN_COLLECT_SCALE))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma
    assert_and_infer_cfg()

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()
    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
    timers['roidb'].toc()
    roidb_size = len(roidb)
    logger.info('{:d} roidb entries'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    # Effective training sample size for one epoch
    train_size = roidb_size // args.batch_size * args.batch_size

    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=args.batch_size,
        drop_last=True
    )
    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)

    ### Model ###
    maskRCNN = Generalized_RCNN()

    if cfg.CUDA:
        maskRCNN.cuda()
        
    ### Optimizer ###
    # record backbone params, i.e., conv_body and box_head params
    gn_params = []
    backbone_bias_params = []
    backbone_bias_param_names = []
    prd_branch_bias_params = []
    prd_branch_bias_param_names = []
    backbone_nonbias_params = []
    backbone_nonbias_param_names = []
    prd_branch_nonbias_params = []
    prd_branch_nonbias_param_names = []
    
    if cfg.MODEL.DECOUPLE:
        for key, value in dict(maskRCNN.named_parameters()).items():
            if not 'so_sem_embeddings.2' in key:
                value.requires_grad = False
                
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'gn' in key:
                gn_params.append(value)
            elif 'Conv_Body' in key or 'Box_Head' in key or 'Box_Outs' in key or 'RPN' in key:
                if 'bias' in key:
                    backbone_bias_params.append(value)
                    backbone_bias_param_names.append(key)
                else:
                    backbone_nonbias_params.append(value)
                    backbone_nonbias_param_names.append(key)
            else:
                if 'bias' in key:
                    prd_branch_bias_params.append(value)
                    prd_branch_bias_param_names.append(key)
                else:
                    prd_branch_nonbias_params.append(value)
                    prd_branch_nonbias_param_names.append(key)
    # Learning rate of 0 is a dummy value to be set properly at the start of training
    params = [
        {'params': backbone_nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': backbone_bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': prd_branch_nonbias_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': prd_branch_bias_params,
         'lr': 0 * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0},
        {'params': gn_params,
         'lr': 0,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY_GN}
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    load_ckpt_dir = './'
    ### Load checkpoint
    if args.load_ckpt_dir:
        load_name = get_checkpoint_resume_file(args.load_ckpt_dir)
        load_ckpt_dir = args.load_ckpt_dir
    elif args.load_ckpt:
        load_name = args.load_ckpt
        load_ckpt_dir = os.path.dirname(args.load_ckpt)

    if args.load_ckpt or args.load_ckpt_dir:
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)

        if cfg.MODEL.DECOUPLE:
            del checkpoint['model']['RelDN.so_sem_embeddings.2.weight']
            del checkpoint['model']['RelDN.so_sem_embeddings.2.bias']
            del checkpoint['model']['RelDN.prd_sem_embeddings.2.weight']
            del checkpoint['model']['RelDN.prd_sem_embeddings.2.bias']

        net_utils.load_ckpt(maskRCNN, checkpoint['model'])
        if args.resume:
            args.start_step = checkpoint['step'] + 1
            if 'train_size' in checkpoint:  # For backward compatibility
                if checkpoint['train_size'] != train_size:
                    print('train_size value: %d different from the one in checkpoint: %d'
                          % (train_size, checkpoint['train_size']))

            # reorder the params in optimizer checkpoint's params_groups if needed
            # misc_utils.ensure_optimizer_ckpt_params_order(param_names, checkpoint)

            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            # optimizer.load_state_dict(checkpoint['optimizer'])
            misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])

        del checkpoint
        torch.cuda.empty_cache()

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    lr = optimizer.param_groups[2]['lr']  # lr of non-backbone parameters, for commmand line outputs.
    backbone_lr = optimizer.param_groups[0]['lr']  # lr of backbone parameters, for commmand line outputs.

    prd_categories = maskRCNN.prd_categories
    obj_categories = maskRCNN.obj_categories
    prd_freq_dict = maskRCNN.prd_freq_dict
    obj_freq_dict = maskRCNN.obj_freq_dict

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True)

    ### Training Setups ###
    args.run_name = misc_utils.get_run_name() + '_step_with_prd_cls_v' + str(cfg.MODEL.SUBTYPE)
    output_dir = misc_utils.get_output_dir(args, args.run_name)
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        ckpt_dir = os.path.join(output_dir, 'ckpt')

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        # if os.path.exists(os.path.join(ckpt_dir, 'best.json')):
        #     best = json.load(open(os.path.join(ckpt_dir, 'best.json')))
        if args.resume and os.path.exists(os.path.join(load_ckpt_dir, 'best.json')):
            logger.info('Loading best json from :' + os.path.join(load_ckpt_dir, 'best.json'))
            best = json.load(open(os.path.join(load_ckpt_dir, 'best.json')))
            json.dump(best, open(os.path.join(ckpt_dir, 'best.json'), 'w'))
        else:
            best = {}
            best['avg_per_class_acc'] = 0.0
            best['iteration'] = 0
            best['accuracies'] = []
            json.dump(best, open(os.path.join(ckpt_dir, 'best.json'), 'w'))

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    args.output_dir = output_dir
    args.do_val = True
    args.use_gt_boxes = True
    args.use_gt_labels = True

    logger.info('Creating val roidb')
    val_dataset_name, val_proposal_file = get_inference_dataset(0)
    val_roidb, val_dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        val_dataset_name, val_proposal_file, None, args.do_val)
    logger.info('Done')


    ### Training Loop ###
    maskRCNN.train()

    # CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)
    # CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER / cfg.TRAIN.SNAPSHOT_FREQ
    CHECKPOINT_PERIOD = 10000
    EVAL_PERIOD = cfg.TRAIN.EVAL_PERIOD
    # Set index for decay steps
    decay_steps_ind = None
    for i in range(1, len(cfg.SOLVER.STEPS)):
        if cfg.SOLVER.STEPS[i] >= args.start_step:
            decay_steps_ind = i
            break
    if decay_steps_ind is None:
        decay_steps_ind = len(cfg.SOLVER.STEPS)

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None)
    try:
        logger.info('Training starts !')
        step = args.start_step
        for step in range(args.start_step, cfg.SOLVER.MAX_ITER):

            maskRCNN.train()
            # Warm up
            if step < cfg.SOLVER.WARM_UP_ITERS:
                method = cfg.SOLVER.WARM_UP_METHOD
                if method == 'constant':
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR
                elif method == 'linear':
                    alpha = step / cfg.SOLVER.WARM_UP_ITERS
                    warmup_factor = cfg.SOLVER.WARM_UP_FACTOR * (1 - alpha) + alpha
                else:
                    raise KeyError('Unknown SOLVER.WARM_UP_METHOD: {}'.format(method))
                lr_new = cfg.SOLVER.BASE_LR * warmup_factor
                net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
                lr = optimizer.param_groups[2]['lr']
                backbone_lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
            elif step == cfg.SOLVER.WARM_UP_ITERS:
                net_utils.update_learning_rate_rel(optimizer, lr, cfg.SOLVER.BASE_LR)
                lr = optimizer.param_groups[2]['lr']
                backbone_lr = optimizer.param_groups[0]['lr']
                assert lr == cfg.SOLVER.BASE_LR

            # Learning rate decay
            if decay_steps_ind < len(cfg.SOLVER.STEPS) and \
                    step == cfg.SOLVER.STEPS[decay_steps_ind]:
                logger.info('Decay the learning on step %d', step)
                lr_new = lr * cfg.SOLVER.GAMMA
                net_utils.update_learning_rate_rel(optimizer, lr, lr_new)
                lr = optimizer.param_groups[2]['lr']
                backbone_lr = optimizer.param_groups[0]['lr']
                assert lr == lr_new
                decay_steps_ind += 1

            training_stats.IterTic()
            optimizer.zero_grad()
            for inner_iter in range(args.iter_size):
                try:
                    input_data = next(dataiterator)
                except StopIteration:
                    dataiterator = iter(dataloader)
                    input_data = next(dataiterator)
                
                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))
                
                net_outputs = maskRCNN(**input_data)
                training_stats.UpdateIterStats(net_outputs, inner_iter)
                loss = net_outputs['total_loss']
                loss.backward()

            optimizer.step()
            training_stats.IterToc()

            training_stats.LogIterStats(step, lr, backbone_lr)

            if (step+1) % EVAL_PERIOD == 0 or (step == cfg.SOLVER.MAX_ITER - 1):
                logger.info('Validating model')
                eval_model = maskRCNN.module
                eval_model = mynn.DataParallel(eval_model, cpu_keywords=['im_info', 'roidb'], device_ids=[0], minibatch=True)
                eval_model.eval()
                all_results = run_eval_inference(eval_model,
                                                 val_roidb,
                                                 args,
                                                 val_dataset,
                                                 val_dataset_name,
                                                 val_proposal_file,
                                                 ind_range=None,
                                                 multi_gpu_testing=False,
                                                 check_expected_results=True)
                csv_path = os.path.join(output_dir, 'eval.csv')
                all_results = all_results[0]
                generate_csv_file_from_det_obj(all_results, csv_path, obj_categories, prd_categories, obj_freq_dict, prd_freq_dict)
                overall_metrics, per_class_metrics = get_metrics_from_csv(csv_path)
                obj_acc = per_class_metrics[(csv_path, 'obj', 'top1')]
                sbj_acc = per_class_metrics[(csv_path, 'sbj', 'top1')]
                prd_acc = per_class_metrics[(csv_path, 'rel', 'top1')]
                avg_obj_sbj = (obj_acc + sbj_acc) / 2.0
                avg_acc = (prd_acc + avg_obj_sbj) / 2.0

                best = json.load(open(os.path.join(ckpt_dir, 'best.json')))
                if avg_acc > best['avg_per_class_acc']:
                    print('Found new best validation accuracy at {:2.2f}%'.format(avg_acc))
                    print('Saving best model..')
                    best['avg_per_class_acc'] = avg_acc
                    best['iteration'] = step
                    best['per_class_metrics'] = {'obj_top1': per_class_metrics[(csv_path, 'obj', 'top1')],
                                    'sbj_top1': per_class_metrics[(csv_path, 'sbj', 'top1')],
                                    'prd_top1': per_class_metrics[(csv_path, 'rel', 'top1')]}
                    best['overall_metrics'] = {'obj_top1': overall_metrics[(csv_path, 'obj', 'top1')],
                            'sbj_top1': overall_metrics[(csv_path, 'sbj', 'top1')],
                            'prd_top1': overall_metrics[(csv_path, 'rel', 'top1')]}
                    save_best_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)
                    json.dump(best, open(os.path.join(ckpt_dir, 'best.json'), 'w'))

            if (step+1) % CHECKPOINT_PERIOD == 0:
                print('Saving Checkpoint..')
                save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)

        # ---- Training ends ----
        # Save last checkpoint
        save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)

    except Exception as e:
        del dataiterator
        logger.info('Save ckpt on exception ...')
        save_ckpt(output_dir, args, step, train_size, maskRCNN, optimizer)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()


if __name__ == '__main__':
    main()
