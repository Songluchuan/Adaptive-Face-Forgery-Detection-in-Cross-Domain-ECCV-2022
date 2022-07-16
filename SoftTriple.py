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

class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K, pl = 0,
                 ori_dim = None, drop_config=None, merge_config=None, 
                 margin_scale=1,
                 positive_weight = [1, 1], push = [0, 0],
                 intra_max = 'softmax', reduction='mean'):
        super(SoftTriple, self).__init__()
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
        self.push = push
        self.intra_max = intra_max
        self.reduction = reduction
        self.positive_weight = torch.tensor(positive_weight).cuda().float()
        if len(self.positive_weight) == self.cN:
            self.loss_fun = F.cross_entropy
        else:
            print('Using custom softmax for the len(positive_weight) != self.cN !!!!!')
            self.loss_fun = self.custom_cross_entropy
        self.world_size = link.get_world_size()
        if ori_dim is not None:
            self.change_dim = nn.Linear(ori_dim, dim)
        else:
            print('Using no dimension change!!!!!')
        self.fc = Parameter(torch.Tensor(dim, sum(self.K)))
        self.weight = torch.zeros(sum(self.K), sum(self.K), dtype=torch.bool).cuda()
        self.register_buffer('_mask', torch.ones(self.fc.size()))
        self.drop_config = drop_config
        self.merge_config = merge_config
        for i in range(0, cN):
            for j in range(0, self.K[i]):
                row = sum(self.K[:i]) + j
                col_end = sum(self.K[:i + 1])
                self.weight[row, row + 1:col_end] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

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
    
    @property
    def mask(self):
        return self._mask

    def drop(self, cur_step):
        if self.drop_config is None:
            return -1
        if cur_step not in self.drop_config.steps:
            valid_kernel_num = self._mask[0].sum().item()
            return valid_kernel_num
        idx = self.drop_config.steps.index(cur_step)
        sim_thr = self.drop_config.sim_thrs[idx]
        tau_weight = self.drop_config.tau_weights[idx]

        centers = F.normalize(self.fc, p=2, dim=0)
        # masking kernel should after normalization
        centers = centers * self._mask
        simCenter = centers.t().matmul(centers)
        refineSimCenter = simCenter * self.weight.float()
        selectedCenter = refineSimCenter > sim_thr
        dropCenter, _ = torch.max(selectedCenter, dim=0)
        appendMask = (1-dropCenter).view([1,-1]).expand_as(self._mask)
        appendMask = appendMask.float()
        self._mask = self._mask * appendMask
        self.tau = self.tau * tau_weight

        valid_kernel_num = self._mask[0].sum().item()
        return valid_kernel_num

    def merge_emb(self, input, target, return_emb = False):
        # the input and centers are NOT normalized!
        assert len(self.merge_config.src_cnt) == len(self.merge_config.target_cnt), "The src_cnt and target_cnt in merge_config must be EQUAL!"
        assert min(self.merge_config.src_cnt) > 1, "The src_cnt in merge_config must be greater than 1!"
        # 1. select indexes based on the src_cnt
        selected_indexes = []
        uniq_label = torch.unique(target)
        # When there are only real embeddings, return 0 
        if len(uniq_label) == 0 and uniq_label[0] == 0:
            return 0 * input.mean()
        for inp_cnt in self.merge_config.src_cnt:
            # when the request cnt <= total label cnt
            if inp_cnt <= uniq_label.numel():
                all_comb_labels = uniq_label.new_tensor(list(
                        itertools.combinations(uniq_label.cpu().numpy(),inp_cnt)))
            # when the request cnt > total label cnt
            else:
                all_comb_labels = [uniq_label.repeat(int(inp_cnt/uniq_label.numel()) + 1)[:inp_cnt]]
            tmp_selected_indexes = []
            for comb_labels in all_comb_labels:
                split_groups = [(target==label).nonzero().reshape(-1) for label in comb_labels]
                tmp_selected_indexes.append(torch.cartesian_prod(*split_groups))
            tmp_selected_indexes = torch.cat(tmp_selected_indexes, dim=0)
            selected_indexes.append(tmp_selected_indexes)
        # 2. random sample combinations based on target_cnt
        sample_selected_indexes = []
        for type_idx, sample_cnt in enumerate(self.merge_config.target_cnt):
            cur_selected_indexes = selected_indexes[type_idx]
            candidate_idx_list = list(range(cur_selected_indexes.size(0)))
            if cur_selected_indexes.size(0) > sample_cnt:
                sample_idx = random.sample(candidate_idx_list,sample_cnt)
            else:
                sample_idx = candidate_idx_list * int(sample_cnt/len(candidate_idx_list) + 1)
                sample_idx = sample_idx[:sample_cnt]
            sample_selected_indexes.append(cur_selected_indexes[sample_idx])
        # 3. merge embeddings
        merged_embs = []
        for group_selected_indexes in sample_selected_indexes:
            for single_selected_indexes in group_selected_indexes:
                num = single_selected_indexes.size(0)
                emb_size = input.size(1)
                mask_idx = list(range(emb_size))
                random.shuffle(mask_idx)
                single_emb_merge_num = int(emb_size/num)
                merged_emb = input.new_zeros(emb_size)
                for emb_idx in range(num):
                    cur_emb = input[single_selected_indexes[emb_idx]]
                    cur_mask = input.new_zeros(emb_size)
                    mask_idx_start = emb_idx * single_emb_merge_num
                    mask_idx_end = (emb_idx + 1) * single_emb_merge_num
                    if emb_idx == num - 1:
                        mask_idx_end = emb_size
                    cur_mask[mask_idx[mask_idx_start:mask_idx_end]] = 1
                    merged_emb = merged_emb + cur_emb * cur_mask
                    merged_embs.append(merged_emb)
        merged_embs = torch.stack(merged_embs,dim=0)
        if return_emb:
            return F.normalize(merged_embs, p=2, dim=1)
        # 4. calculate loss
        merged_embs = F.normalize(merged_embs, p=2, dim=1)
        centers = F.normalize(self.fc, p=2, dim=0)
        # masking kernel should after normalization
        centers = centers * self._mask
        simInd = merged_embs.matmul(centers)

        real_simInd = simInd[:,:self.K[0]]
        fake_simInd = simInd[:,self.K[0]:]
        if self.intra_max == 'max':
            real_simClass, _ = torch.max(real_simInd, dim = 1)
            fake_simClass, _ = torch.max(fake_simInd, dim = 1)
        else:
            real_prob = F.softmax(real_simInd*self.gamma, dim=1)
            real_simClass = torch.sum(real_prob*real_simInd, dim=1)
            fake_prob = F.softmax(fake_simInd*self.gamma, dim=1)
            fake_simClass = torch.sum(fake_prob*fake_simInd, dim=1)
        merged_embs_logit = torch.stack([real_simClass, fake_simClass], dim=1)
        merged_embs_target = merged_embs_logit.new_ones(merged_embs_logit.size(0)).long()
        marginM = merged_embs_logit.new_zeros(merged_embs_logit.shape)
        marginM[torch.arange(0, marginM.shape[0]), merged_embs_target] = self.margin
        loss_merged_embs = self.loss_fun(self.la*(merged_embs_logit-marginM), merged_embs_target, self.positive_weight, reduction=self.reduction)
        loss_merged_embs = loss_merged_embs * self.merge_config.weight
        return loss_merged_embs
        

    def pl_regulation(self, input, target, centers):
        trans_fc = centers.t()
        final_pl = 0
        for idx in range(len(self.K)):
            cur_center = trans_fc[self.K_idx[idx]:self.K_idx[idx+1]]
            valid_idx = (target == target[idx]).nonzero().reshape(-1)
            dist = input[valid_idx, None, :] - cur_center
            min_dist, ind = dist.pow(2).sum(dim = 2).min(dim=1)
            min_dist = min_dist.sum()
            final_pl = final_pl + min_dist
        final_pl = final_pl  / len(target)
        return final_pl
            
        # trans_fc = centers.t().reshape(self.cN, self.K, -1)
        # dist = input[:,None,:] - trans_fc[target]
        # min_dist, ind = dist.pow(2).sum(dim = 2).min(dim=1)
        # return min_dist.mean()

    def set_margin(self, new_margin):
        self.margin = new_margin * self.margin_scale

    def forward(self, input, target, return_class_info=False):
        if self.change_dim is not None:
            input = self.change_dim(input)
        norm_input = F.normalize(input, p=2, dim=1)
        centers = F.normalize(self.fc, p=2, dim=0)
        # masking kernel should after normalization
        centers = centers * self._mask
        simInd = norm_input.matmul(centers)

        # simStruc = simInd.reshape(-1, self.cN, self.K)
        # prob = F.softmax(simStruc*self.gamma, dim=2)
        # simClass = torch.sum(prob*simStruc, dim=2)

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
        simClass = torch.stack(simClass, dim = 1)

        real_target = target
        if len(self.positive_weight) != self.cN:
            real_target = (target > 0).long()
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), real_target] = self.margin
        lossClassify = self.loss_fun(self.la*(simClass-marginM), target, \
                                     self.positive_weight, reduction=self.reduction)
        if return_class_info:
            _, final_class = simClass.topk(1, 1, True, True)
            final_class = final_class.view(-1)
            tmp_simStruc = [simStruc[final_class[i]][i] for i in range(len(final_class))]
            sub_class = []
            for input_logit in tmp_simStruc:
                _, cur_sub_class = input_logit.topk(1, 0, True, True)
                sub_class.append(cur_sub_class)
            sub_class = torch.cat(sub_class, dim=0)
            if self.merge_config is not None:
                merged_embs = self.merge_emb(input, real_target, return_emb = True)
                norm_input = torch.cat([norm_input, merged_embs], dim=0)
            return final_class, sub_class, self.la*simClass, norm_input
        if self.push[0]+self.push[1] > 0 and max(self.K) > 1:
            assert self.reduction=='mean', 'Other loss reductions are not supported for now!'
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.weight.float().sum() * 2)
            push_loss = torch.exp(-reg) * self.push[0]
            if self.training:
                gather_input = self.all_gather(norm_input)
            else:
                gather_input = norm_input
            detach_input = gather_input.detach()
            detach_simInd = detach_input.matmul(centers)
            detach_simStruc = detach_simInd.reshape(-1, sum(self.K))
            detach_logit = F.softmax(detach_simStruc*self.gamma, dim=1)
            max_logit, idx = detach_logit.max(dim = 0)
            max_logit_loss = -torch.log(max_logit).mean() / self.world_size
            push_loss = push_loss + self.push[1] * max_logit_loss
            lossClassify = lossClassify+push_loss
        if self.tau > 0 and max(self.K) > 1:
            assert self.reduction=='mean', 'Other loss reductions are not supported for now!'
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.weight.float().sum() * 2)
            lossClassify = lossClassify+self.tau*reg
        if self.pl > 0:
            assert self.reduction=='mean', 'Other loss reductions are not supported for now!'
            pl_loss = self.pl_regulation(norm_input, real_target, centers)
            lossClassify = lossClassify+self.pl*pl_loss
        if self.merge_config is not None:
            assert self.reduction=='mean', 'Other loss reductions are not supported for now!'
            loss_merged_embs = self.merge_emb(input, real_target)
            lossClassify = lossClassify + loss_merged_embs
        return lossClassify, self.la*simClass
