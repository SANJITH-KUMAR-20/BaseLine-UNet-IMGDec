import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

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


class DiceLoss(nn.Module):

  def __init__(self, epsilon : float = 1e-9):
    super(DiceLoss, self).__init__()
    self.epsilon = epsilon
    
  def forward(self,input : Tensor, target : Tensor, reduce_batch_first : bool = False):
    assert input.size() == target.size()
    #assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1,-2) if input.dim() == 2 or not reduce_batch_first else (-1,-2,-3)

    intersection = 2.0 * ((input * target).sum(dim=sum_dim))
    setssum = (input + target).sum(dim=sum_dim)

    setssum = torch.where(setssum == 0, intersection, setssum)

    dice = (intersection) / (setssum + self.epsilon)
    return 1 - dice.mean()
  
class IOU(nn.Module):
   
   def __init__(self, epsilon : float = 1e-9):
      
      super(IOU,self).__init__()

      self.epsilon = epsilon

   def forward(self, input : Tensor, target : Tensor):
      
      union = (input + target).sum(dim = (-1,-2))
      intersection = (input * target).sum(dim = (-1, -2))

      iou = intersection / (union + self.epsilon)

      return 1 - iou.mean()


class CumLoss(nn.Module):
   
   def __init__(self, core_loss:str = "dice"):
      
      self.iou = IOU()
      self.dice = DiceLoss()
      self.bce = nn.BCEWithLogitsLoss()
      assert core_loss == "dice" or core_loss == "iou"
      self.core_loss = core_loss
   
   def forward(self, input: Tensor, target: Tensor):
      
      loss1 = self.iou(F.sigmoid(input), target.float())
      loss2 = self.bce(input, target.float())
      loss3 = self.dice(F.sigmoid(input), target.float())

      if self.core_loss == "dice":
         
         return (loss3 * 0.5) + (loss2 * 0.5)
      else:
         return (loss1 * 0.5) + (loss2 * 0.5)

      
