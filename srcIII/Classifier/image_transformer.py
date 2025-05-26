import torch                                                                                                                              
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from srcIII import config
from matplotlib.colors import ListedColormap


class Image_Transformer(nn.Module):
    def __init__(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        super(Image_Transformer, self).__init__()

        self.kernels = [
            torch.tensor([[[-1., 1.]]]).unsqueeze(0),
            torch.tensor([[[-1.], [1.]]]).unsqueeze(0),
            torch.tensor([[[0., 1.], [-1., 0.]]]).unsqueeze(0),
            torch.tensor([[[1., 0.], [0., -1.]]]).unsqueeze(0),
        ]

        self.layers = []
        with torch.no_grad():
            for kernel in self.kernels:
                conv_layer = nn.Conv2d(1, 1, kernel.shape[-2:], padding='same', bias=False)
                with torch.no_grad():
                    conv_layer.weight.copy_(kernel)
                conv_layer.requires_grad_(False)
                self.layers.append(conv_layer)
                self.layers.append(nn.ReLU())

            self.conv_block = nn.Sequential(*self.layers)

        self.conv_block.requires_grad_(False)
        self.pool_block = nn.MaxPool2d(kernel_size=6, stride=6)

    def forward(self, x):

        x = self.conv_block(x)
        x = self.pool_block(x)

        x_reshaped = x.view(x.shape[0], -1)
        x_list = x_reshaped.tolist()

        prefix = [config1DImageReal.bos, config1DImageReal.tmp, config1DImageReal.sep]

        x_list_str = []
        for row in x_list:
            row = [int(token) for token in row]
            row_str = list(map(str, row))
            new_row = prefix + row_str
            x_list_str.append(new_row)

        return x_list_str

