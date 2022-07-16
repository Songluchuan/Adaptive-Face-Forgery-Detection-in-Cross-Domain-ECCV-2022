import torch.nn as nn
import math
import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
import torch.nn.functional as F

class OHEMCrossEntropyLoss(_Loss):
    """
    Args:
        topk: the number of hard samples per batch with top loss that will be used for final loss
            and gradient calculation. And the rest are ignored. Default: -1.
        T: the temperature for SoftMax. Default: 1.0
        hack_ratio:
        max_prob_thresh:
    """

    def __init__(self, reduction='mean', topk=-1, T=1.0, hack_ratio=-1.0, min_bp_prob=0.0):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.topk = topk
        self.T = T
        self.hack_ratio = hack_ratio
        self.min_bp_prob = min_bp_prob
        self.reduction = reduction

        # misc.print_with_rank('topk = {}, T = {}, hack_ratio = {}, min_bp_prob = {}'
        #                      .format(self.topk, self.T, self.hack_ratio, self.min_bp_prob), only=0)

    def find_hard_indices(self, input_log_prob, target, num_hns):
        num_inst = input_log_prob.size(0)
        prob = torch.exp(input_log_prob)
        inst_losses = torch.zeros(num_inst).cuda()
        for index, label in enumerate(target):
            inst_losses[index] = -input_log_prob[index, label]
            if prob[index, label] < self.min_bp_prob:
                # misc.print_with_rank('WARNING: Skip a possible outlier '
                #                      '[label: {}, prob: {:0.4f}, loss: {:0.4f}].'
                #                      .format(label, prob[index, label], inst_losses[index]))
                inst_losses[index] = -1.0
        _, indices_hard = inst_losses.topk(num_hns)
        return indices_hard

    def forward(self, input, target):
        with torch.no_grad():
            input_ = input.clone() / self.T
            target_ = target.clone()

            input_log_prob = F.log_softmax(input_, 1)
            num_hns = self.topk if self.topk > 0 else input_log_prob.size(0)

            if self.hack_ratio > 0.0:
                indices_lives = (target_ == 0).nonzero().view(-1)
                indices_hacks = (target_ == 1).nonzero().view(-1)

                num_hns_hacks = int(math.ceil(num_hns * self.hack_ratio))
                num_hns_lives = num_hns - num_hns_hacks

                if num_hns_hacks > indices_hacks.size(0):
                    num_hns_hacks = indices_hacks.size(0)
                    num_hns_lives = num_hns - num_hns_hacks
                    assert indices_lives.size(0) >= num_hns_lives

                if num_hns_lives > indices_lives.size(0):
                    num_hns_lives = indices_lives.size(0)
                    num_hns_hacks = num_hns - num_hns_lives
                    assert indices_hacks.size(0) >= num_hns_hacks

                # live
                input_log_prob_lives = input_log_prob.index_select(0, indices_lives)
                target_lives = target_.index_select(0, indices_lives)
                indices_hard_lives = self.find_hard_indices(input_log_prob_lives, target_lives,
                                                            num_hns_lives)
                indices_hard_lives = torch.Tensor(([indices_lives[i.item()].item()
                                                    for i in indices_hard_lives])).long()
                # hack
                input_log_prob_hacks = input_log_prob.index_select(0, indices_hacks)
                target_hacks = target_.index_select(0, indices_hacks)
                indices_hard_hacks = self.find_hard_indices(input_log_prob_hacks, target_hacks,
                                                            num_hns_hacks)
                indices_hard_hacks = torch.Tensor(([indices_hacks[i.item()].item()
                                                    for i in indices_hard_hacks])).long()
                # concatenate
                indices_hard = torch.cat((indices_hard_lives, indices_hard_hacks))
            else:
                indices_hard = self.find_hard_indices(input_log_prob, target_, num_hns)

        input_selected = input.index_select(0, indices_hard.cuda())
        target_selected = target.index_select(0, indices_hard.cuda())

        return F.cross_entropy(input_selected / self.T, target_selected, weight=None,
                               ignore_index=-100, reduction=self.reduction)


