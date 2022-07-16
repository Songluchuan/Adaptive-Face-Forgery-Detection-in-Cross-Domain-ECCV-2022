import torch
import torch.nn as nn

def CrossEntropyLoss(positive_weight = [1,1]):
    if isinstance(positive_weight, list):
        weight = torch.tensor(positive_weight).float()
    else:
        weight = torch.tensor([1.-positive_weight, positive_weight]) * 2.
    return nn.CrossEntropyLoss(weight=weight)
