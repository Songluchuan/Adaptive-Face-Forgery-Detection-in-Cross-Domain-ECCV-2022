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
import math

class Arcface(nn.Module):
    def __init__(self, la,  margin, dim, cN, 
                 ori_dim = None,  
                 positive_weight = [1, 1], 
                 easy_margin=False,
                 ):
        super(Arcface, self).__init__()
        self.la = la
        self.margin = margin
        self.cN = cN
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.change_dim = None
        self.positive_weight = torch.tensor(positive_weight).cuda().float()
        self.loss_fun = F.cross_entropy
        self.world_size = link.get_world_size()
        if ori_dim is not None:
            self.change_dim = nn.Linear(ori_dim, dim)
        else:
            print('Using no dimension change!!!!!')
        self.fc = Parameter(torch.Tensor(dim, cN))
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        if self.change_dim is not None:
            input = self.change_dim(input)
        norm_input = F.normalize(input, p=2, dim=1)
        centers = F.normalize(self.fc, p=2, dim=0)
        # masking kernel should after normalization
        cosine = norm_input.matmul(centers)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.la # logit for calculating loss
        loss = self.loss_fun(output, target, self.positive_weight)
        return loss, cosine*self.la
