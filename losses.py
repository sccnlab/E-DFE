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



@register
def cross_entropy_loss(
    input: Tensor, 
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    """Cross entropy loss.
    """

    return F.cross_entropy(input, target, weight, size_average, ignore_index, reduce)


@register
def smooth_loss(
    input: Tensor,
    target: Tensor,
    smooth_ratio: float = 0.9,
    weight: Union[None, Tensor] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    '''Smooth loss.
    '''

    prob = F.log_softmax(input, dim=1)
    ymask = prob.data.new(prob.size()).zero_()
    ymask = ymask.scatter_(1, target.view(-1,1), 1)
    ymask = smooth_ratio*ymask + (1-smooth_ratio)*(1-ymask)/(len(input[1])-1)
    loss = - (prob*ymask).sum(1).mean()

    return loss


@register
def multi_smooth_loss(
    input: Tuple,
    target: Tensor,
    smooth_ratio: float = 0.9,
    loss_weight: Union[None, Dict]= None,
    weight: Union[None, Tensor] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    '''Multi smooth loss.
    '''
    assert isinstance(input, tuple), 'input is less than 2'
    print('target size is:', target.size())    
    weight_loss = torch.ones(len(input)).to(input[0].device)
    if loss_weight is not None:
        for item in loss_weight.items():
            weight_loss[int(item[0])] = item[1]

    loss = 0
    for i in range(0, len(input)):
       # print(input[i].size())
       # print(target.size())
        if i in [1, len(input)-1]:
            prob = F.log_softmax(input[i], dim=1)
            ymask = prob.data.new(prob.size()).zero_()
            ymask = ymask.scatter_(1, target.view(-1,1), 1)
            ymask = smooth_ratio*ymask + (1-smooth_ratio)*(1-ymask)/(len(input[i][1])-1)
            loss_tmp = - weight_loss[i]*((prob*ymask).sum(1).mean())
        else:
            loss_tmp = weight_loss[i]*F.cross_entropy(input[i], target, weight, size_average, ignore_index, reduce)
        loss += loss_tmp
    print('Loss shape:', loss)
    return loss
