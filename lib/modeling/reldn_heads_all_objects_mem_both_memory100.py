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
import copy
from .image_encoder_mem import MemoryAugmentedEncoder
from .mem_attention import ScaledDotProductAttentionMemory


logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, n_state=768, n_head=12, n_emb=768):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.c_attn = Conv1D_(n_state * 3, n_state)
        self.c_proj = Conv1D_(n_state, n_state)
        self.split_size = n_state


        # initialize the memory module
        m =100
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, n_state))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, n_state))
        self.m = m

        self.attn_pdrop = nn.Dropout(0.1)

    def _attn(self, q, k, v):

        


        w = torch.matmul(q, k)

        w = nn.Softmax(dim=-1)(w)
        self.w = self.attn_pdrop(w)

        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        b_s, nq = query.shape[:2]



    

        m_k = self.m_k.expand(b_s, self.m, self.split_size)
        m_v = self.m_v.expand(b_s, self.m, self.split_size)


        key = torch.cat([key, m_k],1)
        value = torch.cat([value, m_v],1)



        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a


class Enc_Dec_Attention(nn.Module):
    def __init__(self, n_state=768, n_head =12, n_emb = 768):
        super(Enc_Dec_Attention,self).__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.c_attn = Conv1D_(n_state * 3, n_state)
        self.c_proj = Conv1D_(n_state , n_state)

        self.fc_q = nn.Linear(n_state, n_emb)
        self.fc_k = nn.Linear(n_state, n_emb)
        self.fc_v = nn.Linear(n_state, n_emb)

        self.attn_dropout = nn.Dropout(0.2)
        self.init_weights()
        
    def init_weights(self):

        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)


        
    def _attn(self, q, k, v , enc_dec_attention):

        nk =  k.shape[-1]
        w = torch.matmul(q,k)

        w = w / math.sqrt(v.size(-1))

        nd, ns = w.size(-2), w.size(-1)

        # b = self.bias[-2], w.size(-1)

        if enc_dec_attention is not None:
            w = w.masked_fill(enc_dec_attention, -10000.0)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)



    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states


    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)      


    def forward(self, x, encoder_output=None, mask_encoder=None):

        query = self.fc_q(x)
        encoder_key = self.fc_k(encoder_output)
        encoder_value = self.fc_v(encoder_output)
        query = self.split_heads(query)
        encoder_key = self.split_heads(encoder_key, k=True)
        encoder_value = self.split_heads(encoder_value)


        a = self._attn(query, encoder_key,encoder_value,mask_encoder)
        a = self.merge_heads(a)
        a = self.c_proj(a)


        return a




class MLP(nn.Module):
    def __init__(self, n_state, n_emb):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = n_emb
        self.c_fc = Conv1D_(n_state, nx)
        self.c_proj = Conv1D_(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_state, n_head, n_emb):
        super(Block, self).__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.n_emb = n_emb

        self.ln_1 = LayerNorm(n_emb, eps=1e-5)
        self.attn = Attention(n_state, n_head, n_emb)
        self.ln_2 = LayerNorm(n_emb, eps=1e-5)
        self.mlp = MLP(4 * n_state, n_emb)
        self.resid_pdrop = nn.Dropout(0.1)


        self.enc_dec_attn = Enc_Dec_Attention(n_state, n_head, n_emb)
        self.fc_alpha1 = nn.Linear(n_state + n_state,  n_state)
        self.fc_alpha2 = nn.Linear(n_state+ n_state, n_state)



    def forward(self, x, encoder_features,  mask_encoder):


        #[25, 17, 768]) x shape
        #torch.Size([25, 3, 50, 768]) encoder output shape
        #torch.Size([25, 1, 1, 50]) mask encoder shape
        #torch.Size([25, 17, 768]) a shape

   
        self_attention = self.attn(self.ln_1(x))
        a = x + self_attention

        a = self.resid_pdrop(a)


        enc_att1 = self.enc_dec_attn(x = self.ln_1(a), encoder_output = self.ln_1(encoder_features[:,0]),  mask_encoder = mask_encoder)
        enc_att2 = self.enc_dec_attn(x = self.ln_1(a), encoder_output = self.ln_1(encoder_features[:,1]),  mask_encoder = mask_encoder)

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([a, enc_att1],-1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([a, enc_att2],-1)))

        enc_att1 = alpha1 * a + (1-alpha1) * enc_att1
        enc_att2 = alpha2 * a  + (1-alpha2) * enc_att2

        # threshold = 0.2

        a = (enc_att1  + enc_att2 )/ np.sqrt(2)

        m = self.mlp(self.ln_2(a))

        output = a + m
        output = self.resid_pdrop(output)

        return output


class MultiHeadModel(nn.Module):
    def __init__(self, n_layer, n_state, n_head, n_embd):
        super(MultiHeadModel, self).__init__()
        self.n_layer = n_layer
        self.n_state = n_state
        self.n_head = n_head
        self.n_embd = n_embd

        self.language_fc = nn.Linear(300, n_embd)
        self.visual_fc = nn.Linear(1024, n_embd)

        self.wpe = nn.Embedding(5, n_embd)
        self.wte = nn.Embedding(5, n_embd)
        block = Block(n_state, n_head, n_embd)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layer)])

        self.dropout = nn.Dropout(0.1)

        self.linear_projection = nn.Linear(n_embd, 1024)
        self.layer_norm = nn.LayerNorm(1024, 1e-5)

    def data_transformation(self, sub_label, obj_label, sub_visual, obj_visual, label_visual):
        # print("before")
        # print(sub_label.shape, obj_label.shape, sub_visual.shape, obj_visual.shape, label_visual.shape)
        sub_label = self.language_fc(sub_label)
        sub_label = sub_label.reshape(-1, 1, self.n_embd)
        obj_label = self.language_fc(obj_label)
        obj_label = obj_label.reshape(-1, 1, self.n_embd)

        sub_visual = self.visual_fc(sub_visual)
        sub_visual = sub_visual.reshape(-1, 1, self.n_embd)
        obj_visual = self.visual_fc(obj_visual)
        obj_visual = obj_visual.reshape(-1, 1, self.n_embd)
        label_visual = self.visual_fc(label_visual)
        label_visual = label_visual.reshape(-1, 1, self.n_embd)
        try:
            input_ids = torch.cat([sub_label, obj_label, sub_visual, obj_visual, label_visual], -2)
        except:
            print(sub_label.shape)
            print(obj_label.shape)
            print(sub_visual.shape)
            print(obj_visual.shape)
            print(label_visual.shape)

        position_ids = torch.arange(5, dtype=torch.long, device=sub_label.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])

        position_ids = self.wpe(position_ids)

        type_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=sub_label.device)
        type_ids = type_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])
        type_ids = self.wte(type_ids)

        input_ids = input_ids + position_ids + type_ids
        return input_ids





    def data_transformation_only_visual(self, sub_visual, obj_visual, label_visual):
        # print("before")
        # print(sub_label.shape, obj_label.shape, sub_visual.shape, obj_visual.shape, label_visual.shape)


        sub_visual = self.visual_fc(sub_visual)
        sub_visual = sub_visual.reshape(-1, 1, self.n_embd)
        obj_visual = self.visual_fc(obj_visual)
        obj_visual = obj_visual.reshape(-1, 1, self.n_embd)
        label_visual = self.visual_fc(label_visual)
        label_visual = label_visual.reshape(-1, 1, self.n_embd)
        try:
            input_ids = torch.cat([ sub_visual, obj_visual, label_visual], -2)
        except:

            print(sub_visual.shape)
            print(obj_visual.shape)
            print(label_visual.shape)

        position_ids = torch.arange(3, dtype=torch.long, device=sub_visual.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])

        position_ids = self.wpe(position_ids)

        type_ids = torch.tensor([ 1, 1, 1], dtype=torch.long, device=sub_visual.device)
        type_ids = type_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])
        type_ids = self.wte(type_ids)

        input_ids = input_ids + position_ids + type_ids
        return input_ids



    def forward(self, sub_label, obj_label, sub_visual, obj_visual, label_visual, encoder_features, encoder_mask):
        if sub_label is None:
            hidden_states = self.data_transformation_only_visual(sub_visual, obj_visual, label_visual)
            
          

            for block in self.h:
                hidden_states = block(hidden_states, encoder_features, encoder_mask)

            hidden_states = self.linear_projection(hidden_states)

            hidden_states = self.layer_norm(hidden_states)



            return hidden_states[:, 0, :], hidden_states[:, 1, :], hidden_states[:, 2, :]



        else:
            hidden_states = self.data_transformation(sub_label, obj_label, sub_visual, obj_visual, label_visual)

      
            for block in self.h:    
                hidden_states = block(hidden_states)

            hidden_states = self.linear_projection(hidden_states)

            hidden_states = self.layer_norm(hidden_states)


            return hidden_states[:, 2, :], hidden_states[:, 3, :], hidden_states[:, 4, :]










class reldn_head(nn.Module):
    def __init__(self, dim_in, all_obj_vecs=None, all_prd_vecs=None):
        super().__init__()

        num_prd_classes = cfg.MODEL.NUM_PRD_CLASSES + 1

        if cfg.MODEL.RUN_BASELINE:
            # only run it on testing mode
            self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])
            return

        ### what are these all obj vecs
        self.obj_vecs = all_obj_vecs
        self.prd_vecs = all_prd_vecs

        # add subnet
        self.prd_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1))
        self.prd_vis_embeddings = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))
        # if not cfg.MODEL.USE_SEM_CONCAT:
        #     self.prd_sem_embeddings = nn.Sequential(
        #         nn.Linear(cfg.MODEL.INPUT_LANG_EMBEDDING_DIM, 1024),
        #         nn.LeakyReLU(0.1),
        #         nn.Linear(1024, 1024))
        # else:
        #     self.prd_sem_hidden = nn.Sequential(
        #         nn.Linear(cfg.MODEL.INPUT_LANG_EMBEDDING_DIM, 1024),
        #         nn.LeakyReLU(0.1),
        #         nn.Linear(1024, 1024))
        #     self.prd_sem_embeddings = nn.Linear(3 * 1024, 1024)

        self.prd_sem_embeddings = nn.Sequential(
            nn.Linear(cfg.MODEL.INPUT_LANG_EMBEDDING_DIM, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))

        self.so_vis_embeddings = nn.Linear(dim_in // 3, 1024)
        self.so_sem_embeddings = nn.Sequential(
            nn.Linear(cfg.MODEL.INPUT_LANG_EMBEDDING_DIM, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024))

        if cfg.MODEL.USE_FREQ_BIAS:
            # Assume we are training/testing on only one dataset
            if len(cfg.TRAIN.DATASETS):
                self.freq_bias = FrequencyBias(cfg.TRAIN.DATASETS[0])
            else:
                self.freq_bias = FrequencyBias(cfg.TEST.DATASETS[0])

        self.multi_head_attention = MultiHeadModel(2, 768, 12, 768)

        self.image_encoder =  MemoryAugmentedEncoder(2, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 100})

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

        for p in self.multi_head_attention.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        for p in self.image_encoder.parameters():
            if p.dim() >1:
                nn.init.xavier_uniform_(p)

    def forward(self, spo_feat, sbj_labels=None, obj_labels=None, sbj_feat=None, obj_feat=None,all_unique_features = None):


       

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





        # feed the data into the image encoder 
        enc_output, mask_enc = self.image_encoder(all_unique_features)
        # print(enc_output.shape, "encoder output shape   ")
        # print(mask_enc.shape, "mask encoder shape")






        '''
        
        until here, we can obtain the subject visual embeddings, object visual embeddings, and predicate hidden states
        
        '''
        '''
        the self attention for the sub obj and relation embeddings
        
        '''




           ## get sbj vectors and obj vectors
        sbj_vecs = self.obj_vecs[sbj_labels]  # (#bs, cfg.MODEL.INPUT_LANG_EMBEDDING_DIM)
        sbj_vecs = Variable(torch.from_numpy(sbj_vecs.astype('float32'))).cuda(device_id)

        obj_vecs = self.obj_vecs[obj_labels]  # (#bs, cfg.MODEL.INPUT_LANG_EMBEDDING_DIM)
        obj_vecs = Variable(torch.from_numpy(obj_vecs.astype('float32'))).cuda(device_id)


        sbj_vis_embeddings, obj_vis_embeddings, prd_hidden = self.multi_head_attention(None, None,
                                                                                       sbj_vis_embeddings,
                                                                                       obj_vis_embeddings, prd_hidden, enc_output, mask_enc)





        '''
        all the object and subject word embedding to formalize the object vectors
        '''
        ds_obj_vecs = self.obj_vecs
        ds_obj_vecs = Variable(torch.from_numpy(ds_obj_vecs.astype('float32'))).cuda(device_id)
        so_sem_embeddings = self.so_sem_embeddings(ds_obj_vecs)

        so_sem_embeddings = F.normalize(so_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
        so_sem_embeddings.t_()

        '''
        subject visual embeddings
        '''


        # this is the visual subject embeddings
        sbj_vis_embeddings = F.normalize(sbj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        sbj_sim_matrix = torch.mm(sbj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)

        sbj_cls_scores = cfg.MODEL.NORM_SCALE * sbj_sim_matrix

        # this is the visual object embeddings
        obj_vis_embeddings = F.normalize(obj_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        obj_sim_matrix = torch.mm(obj_vis_embeddings, so_sem_embeddings)  # (#bs, #prd)
        obj_cls_scores = cfg.MODEL.NORM_SCALE * obj_sim_matrix

        '''
        start to predict the predicate features

        '''

        '''
          add self afftention here for subject vis, object vis, prd hidden, subject label, object label

        '''
    

        
        prd_features = torch.cat((sbj_vis_embeddings.detach(), prd_hidden, obj_vis_embeddings.detach()), dim=1)

        prd_vis_embeddings = self.prd_vis_embeddings(prd_features)

        ds_prd_vecs = self.prd_vecs
        ds_prd_vecs = Variable(torch.from_numpy(ds_prd_vecs.astype('float32'))).cuda(device_id)
        prd_sem_embeddings = self.prd_sem_embeddings(ds_prd_vecs)
        prd_sem_embeddings = F.normalize(prd_sem_embeddings, p=2, dim=1)  # (#prd, 1024)
        prd_vis_embeddings = F.normalize(prd_vis_embeddings, p=2, dim=1)  # (#bs, 1024)
        prd_sim_matrix = torch.mm(prd_vis_embeddings, prd_sem_embeddings.t_())  # (#bs, #prd)
        prd_cls_scores = cfg.MODEL.NORM_SCALE * prd_sim_matrix

        if cfg.MODEL.USE_FREQ_BIAS:
            assert sbj_labels is not None and obj_labels is not None
            prd_cls_scores = prd_cls_scores + self.freq_bias.rel_index_with_labels(
                torch.stack((sbj_labels, obj_labels), 1))

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
        return focal_loss.focal_loss(cls_scores_exp, labels_exp, alpha=cfg.MODEL.ALPHA, gamma=cfg.MODEL.GAMMA,
                                     reduction='mean')
    elif cfg.MODEL.LOSS == 'weighted_focal':
        cls_scores_exp = cls_scores.unsqueeze(2)
        cls_scores_exp = cls_scores_exp.unsqueeze(3)
        labels_exp = labels.unsqueeze(1)
        labels_exp = labels_exp.unsqueeze(2)
        weight = weight.unsqueeze(0)
        weight = weight.unsqueeze(2)
        weight = weight.unsqueeze(3)
        return focal_loss.focal_loss(cls_scores_exp, labels_exp, alpha=cfg.MODEL.ALPHA, gamma=cfg.MODEL.GAMMA,
                                     reduction='mean', weight_ce=weight)
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
    hubness_blob = 1. / cls_scores.size(1)
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
    if cfg.MODEL.LOSS == 'weighted_cross_entropy' or cfg.MODEL.LOSS == 'weighted_focal':
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

