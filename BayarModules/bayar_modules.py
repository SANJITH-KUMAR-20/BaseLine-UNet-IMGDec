import torch
import torch.nn as nn
import torch.nn.functional as F



class BayarConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size = 5, stride = 1, padding = 0):
        super(BayarConv, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mid_ele = torch.ones(self.in_channel, self.out_channel, 1) * -1.000

        self.kernel = nn.Parameter(torch.randn((self.in_channel,self.out_channel,kernel_size**2-1)), requires_grad= True)


    def constraint(self):
        self.kernel.data = self.kernel.data.div(self.kernel.data.sum(dim = -1, keepdim=True))
        center = self.kernel_size**2//2
        real_kernel = torch.cat((self.kernel[:,:,:center], self.mid_ele, self.kernel[:,:,center:]),dim=2)
        real_kernel = real_kernel.reshape((self.out_channel, self.in_channel, self.kernel_size, self.kernel_size))
        return real_kernel
    
    def forward(self, x):
        return F.conv2d(x, self.constraint(), stride = self.stride, padding=self.padding)


class BayarConvBlock(nn.Module):

    def __init__(self, in_channel = 12, out_channel = 1024, stride = 1, padding = 0):

        super(BayarConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.stride = stride

        self.mod = nn.Sequential(
            nn.Conv2d(self.in_channel,64,7,stride = 2),
            nn.MaxPool2d(kernel_size= 2),
            nn.ReLU(inplace= True),
            nn.Conv2d(64,128,3,stride = self.stride),
            nn.MaxPool2d(kernel_size= 2),
            nn.ReLU(inplace= True),
            nn.Conv2d(128,256, 5,stride = self.stride),
            nn.MaxPool2d(kernel_size= 2),
            nn.ReLU(inplace= True),
            nn.Conv2d(256,self.out_channel, 3,stride = self.stride),
            nn.MaxPool2d(kernel_size= 2),
            nn.ReLU(inplace= True)
        )

    def forward(self, x):

        return self.mod(x)


class BayarBlock(nn.Module):


    def __init__(self, in_channel = 3, out_channel = 1024, kernel_size = 3, stride = 1, padding = 0):

        super(BayarBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        self.Bayarconv = BayarConv(self.in_channel, 12)
        self.rest = BayarConvBlock(12, self.out_channel)

    def forward(self, x):

        x = self.Bayarconv(x)
        return self.rest(x)