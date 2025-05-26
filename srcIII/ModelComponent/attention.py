import torch
import numpy as np
import torch.nn as nn

class ATT_Module(nn.Module):
    def __init__(self, key_size, num_heads, query_data, key_data, value_data, linear_data, trainable=False):
        super(ATT_Module, self).__init__()

        self.key_size = key_size
        self.num_heads = num_heads

        self.softmax = torch.nn.Softmax(dim=-1)  

        self.Query_Matrix = nn.Linear(query_data['w'].shape[0], query_data['w'].shape[1]) #In features/out features vs out features/in features for the weight matrix
        self.Key_Matrix = nn.Linear(key_data['w'].shape[0], key_data['w'].shape[1])
        self.Value_Matrix = nn.Linear(value_data['w'].shape[0], value_data['w'].shape[1])
        self.Linear_Matrix = nn.Linear(linear_data['w'].shape[0], linear_data['w'].shape[1])

        if not trainable:
            with torch.no_grad():
                self.Query_Matrix.bias.data = torch.tensor(np.array(query_data['b']).copy(), dtype=torch.float32)
                self.Query_Matrix.weight.data = torch.tensor(np.array(query_data['w']).copy(), dtype=torch.float32).T

                self.Key_Matrix.bias.data = torch.tensor(np.array(key_data['b']).copy(), dtype=torch.float32)
                self.Key_Matrix.weight.data = torch.tensor(np.array(key_data['w']).copy(), dtype=torch.float32).T

                self.Value_Matrix.bias.data = torch.tensor(np.array(value_data['b']).copy(), dtype=torch.float32)
                self.Value_Matrix.weight.data = torch.tensor(np.array(value_data['w']).copy(), dtype=torch.float32).T

                self.Linear_Matrix.bias.data = torch.tensor(np.array(linear_data['b']).copy(), dtype=torch.float32)
                self.Linear_Matrix.weight.data = torch.tensor(np.array(linear_data['w']).copy(), dtype=torch.float32).T

            self.Query_Matrix.requires_grad_(False)
            self.Key_Matrix.requires_grad_(False)
            self.Value_Matrix.requires_grad_(False)
            self.Linear_Matrix.requires_grad_(False)
            self.softmax.requires_grad_(False)            
    
    def forward(self, x):
        *leading_dims, _ = x.shape

        query = self.Query_Matrix(x)
        query = query.reshape((*leading_dims, self.num_heads, self.key_size))

        key = self.Key_Matrix(x)
        key = key.reshape((*leading_dims, self.num_heads, self.key_size))

        value = self.Value_Matrix(x)
        value = value.reshape((*leading_dims, self.num_heads, self.key_size))

        attn_logits = torch.einsum("...thd,...Thd->...htT", query, key)
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(np.float32)

        attn_weights = self.softmax(attn_logits) 
        attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value)
        attn = torch.reshape(attn, (*leading_dims, -1))
        
        out = self.Linear_Matrix(attn)
        return x + out
