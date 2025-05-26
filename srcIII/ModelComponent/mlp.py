import torch
import numpy as np
import torch.nn as nn

class MLP_Module(nn.Module):
    def __init__(self, activation_function, linear_1_data, linear_2_data, trainable=False):
        super(MLP_Module, self).__init__()

        self.linear_1 = nn.Linear(linear_1_data['w'].shape[0], linear_1_data['w'].shape[1])
        self.activation_func = activation_function
        self.linear_2 = nn.Linear(linear_2_data['w'].shape[0], linear_2_data['w'].shape[1])  

        if not trainable:  
            with torch.no_grad():
                self.linear_1.bias.data = torch.tensor(np.array(linear_1_data['b']).copy(), dtype=torch.float32)
                self.linear_1.weight.data = torch.tensor(np.array(linear_1_data['w']).copy(), dtype=torch.float32).T

                self.linear_2.bias.data = torch.tensor(np.array(linear_2_data['b']).copy(), dtype=torch.float32)
                self.linear_2.weight.data = torch.tensor(np.array(linear_2_data['w']).copy(), dtype=torch.float32).T

            self.linear_1.requires_grad_(False)
            self.linear_2.requires_grad_(False)
            
    def forward(self, x):
        y = self.linear_1(x)
        y = self.activation_func(y)
        y = self.linear_2(y)
        return x + y
