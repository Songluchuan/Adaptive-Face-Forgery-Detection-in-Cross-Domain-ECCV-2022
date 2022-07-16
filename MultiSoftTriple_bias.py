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
from .SoftTriple_bias import SoftTriple_bias

class MultiSoftTriple_bias(nn.Module):
    def __init__(self, loss_weight, **kwargs):
        super(MultiSoftTriple_bias, self).__init__()
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
            losses.append(SoftTriple_bias(**args))
        self.losses = nn.ModuleList(losses)

    def forward(self, input, target, 
                att_maps=None, return_class_info=False):
        if return_class_info:
            final_class_list = []
            sub_class_list = []
            logit_list = []
            norm_input_list = []
            for idx, loss in enumerate(self.losses):
                cur_input = input[idx]
                cur_target = target[idx]
                att_map = att_maps[idx] if att_maps is not None else None
                final_class, sub_class, cur_logit, norm_input = \
                        loss(cur_input, cur_target, 
                             att_map=att_map, return_class_info=return_class_info)
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
                cur_target = target[idx]
                att_map = att_maps[idx] if att_maps is not None else None
                lossClassify, cur_logit = loss(cur_input, 
                        cur_target, att_map=att_map, 
                        return_class_info=return_class_info)
                final_loss = final_loss + lossClassify * self.loss_weight[idx]
                logit_list.append(cur_logit)
            final_loss = final_loss / sum(self.loss_weight)
            logit_list = torch.stack(logit_list, dim=0)
            return final_loss, logit_list
