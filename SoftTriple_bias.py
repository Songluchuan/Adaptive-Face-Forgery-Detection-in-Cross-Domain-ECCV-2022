import math
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import linklink as link

class SoftTriple_bias(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K, pl = 0,
                 ori_dim = None, regu_config=None, trans_bias=False,
                 margin_scale=1,
                 positive_weight = [1, 1],
                 intra_max = 'softmax', reduction='mean'):
        super(SoftTriple_bias, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin_scale = margin_scale
        self.margin = margin * self.margin_scale
        self.cN = cN
        self.K = K
        if not isinstance(K, list):
            self.K = [K] * cN
        self.K_idx = [0]
        for k in self.K:
            self.K_idx.append(self.K_idx[-1] + k)
        self.change_dim = None
        self.pl = pl
        self.intra_max = intra_max
        self.reduction = reduction
        self.positive_weight = torch.tensor(positive_weight).cuda().float()
        if len(self.positive_weight) == self.cN:
            self.loss_fun = F.cross_entropy
        else:
            print('Using custom softmax for the len(positive_weight) != self.cN !!!!!')
            self.loss_fun = self.custom_cross_entropy
        if ori_dim is not None:
            self.change_dim = nn.Linear(ori_dim, dim)
        else:
            print('Using no dimension change!!!!!')
        if trans_bias:
            self.trans_bias = nn.Linear(int(dim * 5), int(dim*2))
        else:
            self.trans_bias = None
        self.fc = Parameter(torch.Tensor(dim, sum(self.K)))
        self.weight = torch.zeros(sum(self.K), sum(self.K), dtype=torch.bool).cuda()
        self.regu = self.get_regu_fun(regu_config)
        for i in range(0, cN):
            for j in range(0, self.K[i]):
                row = sum(self.K[:i]) + j
                col_end = sum(self.K[:i + 1])
                self.weight[row, row + 1:col_end] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n =  m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
        return

    def get_regu_fun(self, regu_config):
        if regu_config is None:
            return None
        name = regu_config['name']
        weight = regu_config['weight']
        def regu(center, bias):
            c = center.permute(1, 0)
            b = bias.permute(0, 2, 1)
            center_norm = torch.norm(c, dim=-1) # [2,]
            bias_norm = torch.norm(b, dim=-1) # [B,2,]
            if name == 'comp_log':
                return weight*torch.max(bias.new_zeros(1), torch.log(bias_norm / center_norm)).mean()
            elif name == 'comp_line':
                return weight*torch.max(bias.new_zeros(1), bias_norm - center_norm).mean()
            elif name == 'comp_exp':
                return weight*torch.max(bias.new_zeros(1), torch.exp(bias_norm - center_norm)).mean()
            elif  name == 'norm':
                return weight*bias_norm.mean()
        return regu
                
    def custom_cross_entropy(self, input, target, weight, reduction):
        real_target = (target > 0).long()
        batch_loss = F.cross_entropy(input, real_target, reduction='none') # [batch]
        batch_weight = weight[target]
        if reduction=='mean':
            loss = (batch_loss * batch_weight).sum() / (batch_weight.sum() + 1e-9)
        elif reduction=='none':
            # Better not use this
            loss = batch_loss * batch_weight / (batch_weight.sum() + 1e-9)
        return loss
    
    def pl_regulation(self, input, target, centers):
        if len(centers.size()) == 2:
            centers = centers[None, :, :].repeat(input.size(0),1,1)
        # trans_fc = centers.permute(0,2,1)
        trans_fc = centers
        final_pl = 0
        for idx in range(len(self.K)):
            cur_center = trans_fc[:, self.K_idx[idx]:self.K_idx[idx+1]]
            # print(cur_center.size())
            valid_idx = (target == target[idx]).nonzero().reshape(-1)
            # print(input[valid_idx, None, :].size(), cur_center[valid_idx].size())
            dist = input[valid_idx, None, :] - cur_center[valid_idx]
            min_dist, ind = dist.pow(2).sum(dim = 2).min(dim=1)
            min_dist = min_dist.sum()
            final_pl = final_pl + min_dist
        final_pl = final_pl  / len(target)
        return final_pl
            
    def forward(self, input, target, bias=None, margin=None, 
                att_map=None, return_class_info=False):
        original_input_size = input.size()
        if len(original_input_size) == 3: # [B, H*W, C]
            input = input.reshape(-1, original_input_size[-1]) # [B, H*W, C] -> [B*H*W, C]
        # print('input', input.size())
        if margin is not None:
            self.margin = margin * self.margin_scale
        if self.change_dim is not None:
            input = self.change_dim(input)
        # print('input_dim', input.size())
        cur_fc = self.fc
        norm_input = F.normalize(input, p=2, dim=1)
        if bias is not None:
            assert len(original_input_size) == 2, 'bias NOT support 3d input now!'
            if self.trans_bias is not None:
                batch_centers = cur_fc.reshape(1, -1).repeat(bias.size(0), 1) # [128,2] -> [B, 256]
                bias_input_feature = torch.cat([input, bias, batch_centers], dim=1) # [B, 640]
                bias = self.trans_bias(bias_input_feature) # [B, 256]
            bias = bias.reshape(bias.size(0), self.fc.size(0), self.fc.size(1))
            centers = cur_fc + bias
            centers = centers.permute(0, 2, 1) # [16, 128, 2] -> [16, 2, 128]
            centers = F.normalize(centers, p=2, dim=-1)
            simInd = centers * norm_input[:,None,:] # [16, 2, 128] * [16, 1, 128]
            simInd = simInd.sum(dim = -1) # [16, 2]
        else:
            centers = F.normalize(cur_fc, p=2, dim=0)
            simInd = norm_input.matmul(centers)
        # print('simInd', simInd.size())

        simClass = []
        simStruc = []
        for idx in range(len(self.K)):
            cur_simStruc = simInd[:, self.K_idx[idx]:self.K_idx[idx+1]] # [Batch, K[idx]]
            if self.intra_max == 'max':
                cur_simClass, _ = torch.max(cur_simStruc, dim=1) # [Batch, ]
            else:
                cur_prob = F.softmax(cur_simStruc*self.gamma, dim=1)
                cur_simClass = torch.sum(cur_prob*cur_simStruc, dim=1) # [Batch, ]
            simClass.append(cur_simClass)
            simStruc.append(cur_simStruc)
        simClass = torch.stack(simClass, dim = 1) # [Batch, 2]
        # print('simClass', simClass.size())
        # merge 3d logit to 2d
        if len(original_input_size) == 3: # [B, H*W, C]
            simClass = simClass.reshape(original_input_size[0], original_input_size[1], -1) # [B*H*W, 2] -> [B, H*W, 2]
            if att_map is None:
                simClass = simClass.mean(dim=1) # [B, H*W, 2] -> [B, 2]
            else:
                att_map = att_map.reshape(att_map.size(0), att_map.size(1), -1) # [B,C,H,W] -> [B,C,H*W]
                att_map = att_map.permute(0, 2, 1) # [B,C,H*W] -> [B,H*W,C]
                simClass = (simClass * att_map).sum(dim=1) # [B, H*W, 2] -> [B, 2]
            # print('simClass_trans', simClass.size())

        real_target = target
        if len(self.positive_weight) != self.cN:
            real_target = (target > 0).long()
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), :] = self.margin
        lossClassify = self.loss_fun(self.la*(simClass-marginM), target, \
                                     self.positive_weight, reduction=self.reduction)
        if return_class_info:
            assert len(original_input_size) == 2, 'bias NOT support 3d input now!'
            _, final_class = simClass.topk(1, 1, True, True)
            final_class = final_class.view(-1)
            tmp_simStruc = [simStruc[final_class[i]][i] for i in range(len(final_class))]
            sub_class = []
            for input_logit in tmp_simStruc:
                _, cur_sub_class = input_logit.topk(1, 0, True, True)
                sub_class.append(cur_sub_class)
            sub_class = torch.cat(sub_class, dim=0)
            return final_class, sub_class, self.la*simClass, norm_input
        if self.tau > 0 and max(self.K) > 1:
            assert self.reduction=='mean', 'Other loss reductions are not supported for now!'
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.weight.float().sum() * 2)
            lossClassify = lossClassify+self.tau*reg
        if self.pl > 0:
            assert len(original_input_size) == 2, 'bias NOT support 3d input now!'
            assert self.reduction=='mean', 'Other loss reductions are not supported for now!'
            pl_loss = self.pl_regulation(norm_input, real_target, centers)
            lossClassify = lossClassify+self.pl*pl_loss
        if self.regu is not None and bias is not None:
            regu_loss = self.regu(self.fc, bias)
            # print(lossClassify, regu_loss)
            lossClassify = lossClassify+regu_loss
        return lossClassify, self.la*(simClass-marginM)

if __name__=="__main__":
    model = SoftTriple_bias(20, 10, 0, 0, 128, 2, 1, ori_dim=2048)
    model.eval()
    params = list(model.named_parameters())
    for name, param in params:
        print(name)
        # print(param)
