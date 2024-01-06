import torch
import torch.nn as nn
import torch.nn.functional as F
from U_net_Modules import *


class UNet(nn.Module):

    def __init__(self, inp_channels, out_channels):

        super(UNet,self).__init__()

        self.in_channel = inp_channels
        self.out_channels = out_channels

        self.initial = ConvBlock(inp_channels, 64)
        self.contract1 = EncodeDown(64, 128)
        self.contract2 = EncodeDown(128, 256)
        self.contract3 = EncodeDown(256, 512)
        self.contract4 = EncodeDown(512, 1024)
        self.expand1 = DecodeUp(1024,512)
        self.expand2 = DecodeUp(512, 256)
        self.expand3 = DecodeUp(256, 128)
        self.expand4 = DecodeUp(128, 64)
        self.resConv = ResultConv(64, out_channels)

    def forward(self, x):

        x1 = self.initial(x)
        x2 = self.contract1(x1)
        x3 = self.contract2(x2)
        x4 = self.contract3(x3)
        x5 = self.contract4(x4)
        x = self.expand1(x5, x4)
        x = self.expand2(x, x3)
        x = self.expand3(x, x2)
        x = self.expand4(x, x1)
        x = self.resConv(x)
        
        return x
    
    def use_checkpointing(self):
        self.initial = torch.utils.checkpoint(self.initial)
        self.contract1 = torch.utils.checkpoint(self.contract1)
        self.contract2 = torch.utils.checkpoint(self.contract2)
        self.contract3 = torch.utils.checkpoint(self.contract3)
        self.contract4 = torch.utils.checkpoint(self.contract4)
        self.expand1 = torch.utils.checkpoint(self.expand1)
        self.expand2 = torch.utils.checkpoint(self.expand2)
        self.expand3 = torch.utils.checkpoint(self.expand3)
        self.expand4 = torch.utils.checkpoint(self.expand4)
        self.resConv = torch.utils.checkpoint(self.resConv)
