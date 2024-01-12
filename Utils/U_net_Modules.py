import numpy as np
import torch
import torchvision
import cv2 as cv
import torch.nn.functional as F
import torch.nn as nn


class ResNetBlock(nn.Module):

    def __init__(self, Top_rem : int = -2):
        super().__init__(self, ResNetBlock)
        
        self.ResBlock = torchvision.models.resnet152(pretrained = False)
        self.applied = nn.Sequential(*list(self.ResBlock.children())[:Top_rem])

    def forward(self, x):
        return self.applied(x)

class ConvBlock(nn.Module):

    def __init__(self, in_channel,out_channel, mid_channel = None, isBias = False):
        super(ConvBlock,self).__init__()

        if not mid_channel:
            mid_channel = out_channel

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels= mid_channel, kernel_size = 3, stride = 1, padding= 1, bias = isBias),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels= out_channel, kernel_size = 3, stride = 1, padding= 1, bias = isBias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.dconv(x)
    
class EncodeDown(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(EncodeDown,self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size= 2),
            ConvBlock(in_channel, out_channel)
        )


    def forward(self, x):
        return self.down(x)
    
class DecodeUp(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DecodeUp,self).__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners= True)
        self.dconv = ConvBlock(in_channel, out_channel, in_channel//2)

    def forward(self, x, y):
        
        x = self.up(x)

        HFactor = abs(y.size()[1] - x.size()[1])
        WFactor = abs(y.size()[2] - x.size()[2])

        x = F.pad(x , [WFactor//2, WFactor - WFactor//2, HFactor//2, HFactor - HFactor//2])

        x = torch.cat([y, x], dim = 1)
        return self.dconv(x)


class ResultConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ResultConv,self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size= 1)

    def forward(self, x):
        return self.conv(x)
    
class BayarConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, stride, padding):

        self.in_channel = in_channel
        self.out_channel = out_channel

# class SkipBlock(nn.Module):

#     def __init__(self, in_channel, out_channel):