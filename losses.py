from typing import Union, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from nest import register


@register
def prop_mse(
    input: Tuple,
    target: Tensor) -> Tensor:
    """Custom PSME Loss.
    """
    final_loss = 0
    for i in range(0,len(input)):
        pred = input[i]
        total = target.size()[0]*target.size()[1]
        target = target.reshape(-1,1)
        pred = pred.reshape(-1,1)
        nonzeros = len(torch.nonzero(target))
        zeros = len(torch.nonzero(target==0))
        zero_weight = (1-(zeros/total))
        nonzero_weight = (1-(nonzeros/total))
        unweighted_error = (target - pred)**2
        if len(torch.nonzero(target)) == 0:
           active_error = 0
        else:
           active_error = torch.sum(unweighted_error[torch.nonzero(target).long()[:,0]]*nonzero_weight)   
        inactive_error = torch.sum(unweighted_error[torch.nonzero(target==0).long()[:,0]]*zero_weight)
        loss = active_error + inactive_error
        final_loss += loss

    return final_loss



