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

class MultiST_NL_Weight(nn.Module):
    def __init__(self, **kwargs):
        super(MultiST_NL_Weight, self).__init__()
        losses = []
        plen = len(kwargs['positive_weight'])
        for key in kwargs:
            if not isinstance(kwargs[key], list):
                kwargs[key] = [kwargs[key]] * plen
        for i in range(plen):
            args = {}
            for key in kwargs:
                args[key] = kwargs[key][i]
            losses.append(SoftTriple(**args))
        self.losses = nn.ModuleList(losses)

    def forward(self, head_features, nl_logits, target, return_class_info=False):
        nl_logits = torch.cat(nl_logits, dim=1)
        nl_weight = F.softmax(nl_logits, dim=1)
        if return_class_info:
            final_class_list = []
            sub_class_list = []
            logit_list = []
            norm_input_list = []
            for idx, loss in enumerate(self.losses):
                cur_input = head_features[idx]
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
            return final_class_list, sub_class_list, logit_list, norm_input_list, nl_weight
        else:
            loss_list = []
            logit_list = []
            for idx, loss in enumerate(self.losses):
                cur_input = head_features[idx]
                # squeeze features
                if len(cur_input.shape) > 2:
                    cur_input = F.adaptive_avg_pool2d(cur_input, (1, 1))
                    cur_input = cur_input.view(cur_input.size(0), -1)
                cur_target = target[idx]
                lossClassify, cur_logit = loss(cur_input, cur_target, return_class_info)
                loss_list.append(lossClassify)
                logit_list.append(cur_logit)
            stack_loss = torch.stack(loss_list, dim=1)
            stack_logit = torch.stack(logit_list, dim=1)
            final_loss = (stack_loss * nl_weight).sum(dim = 1).mean()
            final_logit = (stack_logit * nl_weight[:,:,None]).sum(dim=1)
            return final_loss, final_logit
