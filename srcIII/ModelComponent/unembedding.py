import torch
import numpy as np
import torch.nn as nn

class Unembedding_Module(nn.Module):
    def __init__(self, unembedding_data, use_unembed_argmax, trainable=False):
        super(Unembedding_Module, self).__init__()

        self.use_unembed_argmax = use_unembed_argmax
        self.unembed = nn.Linear(unembedding_data.shape[0], unembedding_data.shape[1], bias=False)

        if not trainable:
            with torch.no_grad():
                self.unembed.weight.data = torch.tensor(np.array(unembedding_data).copy(), dtype=torch.float32).T

            self.unembed.requires_grad_(False)
            
    def forward(self, x):
        x = self.unembed(x)
        if(self.use_unembed_argmax):
            x = torch.argmax(x, axis=-1)
        return x