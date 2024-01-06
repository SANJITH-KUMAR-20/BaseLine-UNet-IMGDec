import torch 
from torch import Tensor

def Dice_Ceff(input : Tensor, target : Tensor, reduce_batch_first : bool = False, epsilon : float = 1e-6):

    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1,-2) if input.dim() == 2 or not reduce_batch_first else (-1,-2,-3)

    intersection = 2 * (input * target).sum(dim=sum_dim)
    setssum = input.sum(dim=sum_dim) + target.sum(dim = sum_dim)

    setssum = torch.where(setssum == 0, intersection, setssum)

    dice = (intersection + epsilon) / (setssum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor):
    dice_score = Dice_Ceff(input, target, reduce_batch_first=True)
    return 1 - dice_score