# ------------------------------------------------------------------------------
# Copyright (c) SenseTime
# Written by Joey Fang (fangzheng@sensetime.com)
# ------------------------------------------------------------------------------

from .LabelSmoothCELoss import LabelSmoothCELoss
from .CrossEntropyLoss import CrossEntropyLoss
from .SoftTriple import SoftTriple
from .MultiSoftTriple import MultiSoftTriple
from .MultiSoftTriple_bias import MultiSoftTriple_bias
from .MultiST_NL_Weight import MultiST_NL_Weight
from .MultiST_Margin import MultiST_Margin
from .SoftTriple_bias import SoftTriple_bias
from .Arcface import Arcface
import torch.nn as nn

__all__ = ['LabelSmoothCELoss', 'CrossEntropyLoss', 'SoftTriple', 'MultiSoftTriple', 'MultiST_NL_Weight', 'MultiST_Margin', 
           'SoftTriple_bias', 'MultiSoftTriple_bias', 'Arcface']

def loss_entry(config):
    loss_name = config['type']
    kwargs = config['kwargs']
    # 1. check wether this loss_fun has been defined in pfa
    if loss_name in __all__:
        return globals()[loss_name](**kwargs)
    # 2. find this loss_fun in pytorch
    else:
        loss_fun = getattr(nn, loss_name, None)
        assert loss_fun is not None, '{} is NOT supported by either PFA or PyTorch.'.format(loss_name)
        return loss_fun(**kwargs)
        
