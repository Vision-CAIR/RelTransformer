import torch
import torch.nn as nn
from modeling.cos_norm_classifier import CosNorm_Classifier
from utils.memory_utils import *

import pdb

class MetaEmbedding_Classifier(nn.Module):
    
    def __init__(self, feat_dim=2048, num_classes=1000):
        super(MetaEmbedding_Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc_hallucinator = nn.Linear(feat_dim, num_classes)
        self.fc_selector = nn.Linear(feat_dim, feat_dim)
        self.cosnorm_classifier = CosNorm_Classifier(feat_dim, num_classes)
        
    def forward(self, x, centroids, *args):
        
        # storing direct feature
        direct_feature = x

        batch_size = x.size(0)
        feat_size = x.size(1)
        
        # set up visual memory
        #x_expand = x.unsqueeze(1).expand(-1, self.num_classes, -1)
        #centroids_expand = centroids.unsqueeze(0).expand(batch_size, -1, -1)
        keys_memory = centroids
        #print(x_expand.shape)
        #print(centroids_expand.shape)
        # computing reachability
        #dist_cur = torch.norm(x_expand - centroids_expand, 2, 2)
        #values_nn, labels_nn = torch.sort(dist_cur, 1)
        #scale = 10.0
        #reachability = (scale / values_nn[:, 0]).unsqueeze(1).expand(-1, feat_size)

        # computing memory feature by querying and associating visual memory
        values_memory = self.fc_hallucinator(x)
        values_memory = values_memory.softmax(dim=1)
        memory_feature = torch.matmul(values_memory, keys_memory)

        # computing concept selector
        concept_selector = self.fc_selector(x)
        concept_selector = concept_selector.tanh() 
        #x = reachability * (direct_feature + concept_selector * memory_feature)
        x = (direct_feature + concept_selector * memory_feature)

        # storing infused feature
        infused_feature = concept_selector * memory_feature
        
        logits = self.cosnorm_classifier(x)

        return logits, [direct_feature, infused_feature]
    
def create_model(feat_dim=2048, num_classes=1000, stage1_weights=False, dataset=None, test=False, prd=False, *args):
    print('Loading Meta Embedding Classifier.')
    clf = MetaEmbedding_Classifier(feat_dim, num_classes)
    weights = 'Outputs/e2e_relcnn_VGG16_8_epochs_gvqa_y_loss_only_1_gpu/gvqa/Feb07-10-55-03_login104-09_step_with_prd_cls_v3/ckpt/model_step1439.pth'
    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc_hallucinator = init_weights(model=clf.fc_hallucinator,
                                                    weights_path=weights,
                                                    classifier=True,
                                                    prd=prd)
        else:
            print('Random initialized classifier weights.')

    return clf
