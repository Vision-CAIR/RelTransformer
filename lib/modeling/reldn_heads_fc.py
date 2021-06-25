# modified by Sherif Abdelkarim on Jan 2020

import numpy as np
from numpy import linalg as la
import math
import logging
import json

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import nn as mynn

from core.config import cfg
import utils.net as net_utils
from modeling.sparse_targets_rel import FrequencyBias
from utils import focal_loss
from .transformer import LayerNorm, Conv1D_, gelu

logger = logging.getLogger(__name__)


class reldn_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
            
        num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1

        # add subnet
        self.prd_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1))
        self.prd_vis_embeddings = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))

        self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)

        self.classifier_prd = nn.Linear(1024, num_prd_classes)
        self.classifier_sbj_obj = nn.Linear(1024, cfg.MODEL.NUM_CLASSES - 1)
        self._init_weights()

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # spo_feat is concatenation of SPO
    def forward(self, spo_feat, sbj_labels=None, obj_labels=None, sbj_feat=None, obj_feat=None):

        device_id = spo_feat.get_device()
        if sbj_labels is not None:
            sbj_labels = Variable(torch.from_numpy(sbj_labels.astype('int64'))).cuda(device_id)
        if obj_labels is not None:
            obj_labels = Variable(torch.from_numpy(obj_labels.astype('int64'))).cuda(device_id)
            
        if cfg.MODEL.RUN_BASELINE:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = self.freq_bias.rel_index_with_labels(torch.stack((sbj_labels, obj_labels), 1))
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
            return prd_cls_scores, None, None, None, None, None
        
        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)
        
        sbj_vis_embeddings = self.so_vis_embeddings(sbj_feat)
        obj_vis_embeddings = self.so_vis_embeddings(obj_feat)
        
        prd_hidden = self.prd_feats(spo_feat)
        prd_features = torch.cat((sbj_vis_embeddings.detach(), prd_hidden, obj_vis_embeddings.detach()), dim=1)
        prd_vis_embeddings = self.prd_vis_embeddings(prd_features)

        sbj_vis_embeddings = F.normalize(sbj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        sbj_cls_scores = self.classifier_sbj_obj(sbj_vis_embeddings)

        obj_vis_embeddings = F.normalize(obj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        obj_cls_scores = self.classifier_sbj_obj(obj_vis_embeddings)

        prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        prd_cls_scores = self.classifier_prd(prd_vis_embeddings)

        if not self.training:
            sbj_cls_scores = F.softmax(sbj_cls_scores, dim=1)
            obj_cls_scores = F.softmax(obj_cls_scores, dim=1)
            prd_cls_scores = F.softmax(prd_cls_scores, dim=1)
        
        return prd_cls_scores, sbj_cls_scores, obj_cls_scores


def add_cls_loss(cls_scores, labels, weight=None):
    if cfg.MODEL.LOSS == 'cross_entropy':
        return F.cross_entropy(cls_scores, labels)
    elif cfg.MODEL.LOSS == 'weighted_cross_entropy':
        return F.cross_entropy(cls_scores, labels, weight=weight)
    elif cfg.MODEL.LOSS == 'focal':
        cls_scores_exp = cls_scores.unsqueeze(2)
        cls_scores_exp = cls_scores_exp.unsqueeze(3)
        labels_exp = labels.unsqueeze(1)
        labels_exp = labels_exp.unsqueeze(2)
        return focal_loss.focal_loss(cls_scores_exp, labels_exp, alpha=cfg.MODEL.ALPHA, gamma=cfg.MODEL.GAMMA, reduction='mean')
    else:
        raise NotImplementedError

def add_hubness_loss(cls_scores):
    # xp_yall_prob   (batch_size, num_classes)
    # xp_yall_prob.T (num_classes, batch_size
    # xp_yall_prob.expand(0, 1, -1, 1)
    # xp_yall_probT_average_reshape = xp_yall_probT_reshaped.mean(axis=2)
    # hubness_dist = xp_yall_probT_average_reshape - hubness_blob
    # hubness_dist_sqr = hubness_dist.pow(2)
    # hubness_dist_sqr_scaled = hubness_dist_sqr * cfg.TRAIN.HUBNESS_SCALE
    cls_scores = F.softmax(cls_scores, dim=1)
    hubness_blob = 1./cls_scores.size(1)
    cls_scores_T = cls_scores.transpose(0, 1)
    cls_scores_T = cls_scores_T.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, 1)
    cls_scores_T = cls_scores_T.mean(dim=2, keepdim=True)
    hubness_dist = cls_scores_T - hubness_blob
    hubness_dist = hubness_dist.pow(2) * cfg.TRAIN.HUBNESS_SCALE
    hubness_loss = hubness_dist.mean()
    return hubness_loss


def reldn_losses(prd_cls_scores, prd_labels_int32, fg_only=False, weight=None):
    device_id = prd_cls_scores.get_device()
    prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).cuda(device_id)
    if cfg.MODEL.LOSS == 'weighted_cross_entropy':
        weight = Variable(torch.from_numpy(weight)).cuda(device_id)
    loss_cls_prd = add_cls_loss(prd_cls_scores, prd_labels, weight=weight)
    # class accuracy
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss_cls_prd, accuracy_cls_prd


def reldn_so_losses(sbj_cls_scores, obj_cls_scores, sbj_labels_int32, obj_labels_int32):
    device_id = sbj_cls_scores.get_device()

    sbj_labels = Variable(torch.from_numpy(sbj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_sbj = add_cls_loss(sbj_cls_scores, sbj_labels)
    sbj_cls_preds = sbj_cls_scores.max(dim=1)[1].type_as(sbj_labels)
    accuracy_cls_sbj = sbj_cls_preds.eq(sbj_labels).float().mean(dim=0)
    
    obj_labels = Variable(torch.from_numpy(obj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_obj = add_cls_loss(obj_cls_scores, obj_labels)
    obj_cls_preds = obj_cls_scores.max(dim=1)[1].type_as(obj_labels)
    accuracy_cls_obj = obj_cls_preds.eq(obj_labels).float().mean(dim=0)
    
    return loss_cls_sbj, loss_cls_obj, accuracy_cls_sbj, accuracy_cls_obj
