import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNetInterMediateResult(nn.Module):

    def __init__(self, model : torchvision.models.resnet.ResNet = torchvision.models.resnet50(pretrained = True)):
        super(ResNetInterMediateResult,self).__init__()
        
        self.model = model
        self.layers = nn.Sequential(*list(self.model.children())[:-2]) 
        self.layer_check = []
        self.inter_results = []

    def forward(self, x):
        for layer_name, layer in self.layers._modules.items():
            x = layer(x)
            self.layer_check.append(layer)
            self.inter_results.append(x)

        return self.inter_results
    

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, padding = (1,1), stride = (1,1)):
        super(UpsampleBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride

        self.up = nn.Upsample(scale_factor= 2, mode = "bilinear")
        self.up_channel = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,kernel_size=(3,3), stride = stride, padding = padding)

    def forward(self, x):
        x = self.up(x)
        x = self.up_channel(x)
        return x
    

class Upsample_and_concat(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Upsample_and_concat,self).__init__()

        self.upsample1 = UpsampleBlock(in_channels=in_channels, out_channels=1024)
        self.upsample2 = UpsampleBlock(1024, 512)
        self.upsample3 = UpsampleBlock(512, 256)
        self.upsample4 = UpsampleBlock(256, 128)
        self.upsample5 = UpsampleBlock(128,out_channels)
        

    def _concat(self, x, y):
        return torch.cat((x[:,:x.shape[1]//2,:,:],y[:, :y.shape[1]//2, :, :]),dim = 1)

        
    def forward(self, x):

        x1 = self.upsample1(x[-1])
        res1 = self._concat(x[-2], x1)
        x2 = self.upsample2(res1)
        res2 = self._concat(x[-3],x2)
        x3 = self.upsample3(res2)
        res3 = self._concat(x[-4], x3)
        x4 = self.upsample4(res3)
        x5 = self.upsample5(x4)

        return x5
