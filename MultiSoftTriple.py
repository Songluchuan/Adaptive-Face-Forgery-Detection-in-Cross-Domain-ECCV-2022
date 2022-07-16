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
from .SoftTriple import SoftTriple

class MultiSoftTriple(nn.Module):
    def __init__(self, loss_weight, **kwargs):
        super(MultiSoftTriple, self).__init__()
        self.loss_weight = loss_weight
        losses = []
        plen = len(loss_weight)
        for key in kwargs:
            if not isinstance(kwargs[key], list):
                kwargs[key] = [kwargs[key]] * plen
        for i in range(plen):
            args = {}
            for key in kwargs:
                args[key] = kwargs[key][i]
            losses.append(SoftTriple(**args))
        self.losses = nn.ModuleList(losses)

    def forward(self, input, target, return_class_info=False):
        if return_class_info:
            final_class_list = []
            sub_class_list = []
            logit_list = []
            norm_input_list = []
            for idx, loss in enumerate(self.losses):
                cur_input = input[idx]
                # squeeze features
                if len(cur_input.shape) > 2:
                    cur_input = F.adaptive_avg_pool2d(cur_input, (1, 1))
                    cur_input = cur_input.view(cur_input.size(0), -1)
                cur_target = target[idx]
                final_class, sub_class, cur_logit, norm_input = \
                        loss(cur_input, cur_target, return_class_info)
                final_class_list.append(final_class)
                sub_class_list.append(sub_class)
                logit_list.append(cur_logit)
                norm_input_list.append(norm_input)
            return final_class_list, sub_class_list, logit_list, norm_input_list
        else:
            final_loss = 0
            logit_list = []
            for idx, loss in enumerate(self.losses):
                cur_input = input[idx]
                # squeeze features
                if len(cur_input.shape) > 2:
                    cur_input = F.adaptive_avg_pool2d(cur_input, (1, 1))
                    cur_input = cur_input.view(cur_input.size(0), -1)
                cur_target = target[idx]
                lossClassify, cur_logit = loss(cur_input, cur_target, return_class_info)
                final_loss = final_loss + lossClassify * self.loss_weight[idx]
                logit_list.append(cur_logit)
            final_loss = final_loss / sum(self.loss_weight)
            logit_list = torch.stack(logit_list, dim=0)
            return final_loss, logit_list
