import torch.nn.functional as F
from torch import nn

class Model_1D_trainable(nn.Module):
    def __init__(self, model_config, model_parameters, unembedding_data, encoding_func, encoded_vocab, config):
        super(Model_1D_trainable, self).__init__()

        self.embedding_dim = unembedding_data.shape[0]

        self.encoder = encoding_func

        self.embedder = Embedding_Module(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings'],
                                                                  trainable=True).to(config.device)

        self.classifier = StaticTokenReplacer(token_embeddings=self.embedder.token_embed.weight.data, pos_embed_layer=self.embedder.pos_embed)
        
        self.transformer = Transformer_Module(model_config=model_config,
                                                                         model_parameters=model_parameters,
                                                                         trainable=True,
                                                                         )
        
        self.unembedder = Unembedding_Module(unembedding_data=unembedding_data,
                                                                        use_unembed_argmax=False,
                                                                        trainable=True)
            
    def forward(self, x, temperature=0.1, hard=False):
        x = self.encoder(x)        
        x = self.embedder(x)
        x = self.classifier(x, temperature=temperature, hard=False)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
    def foward_without_classification(self, x):
        x = self.encoder(x)    
        x = self.embedder(x)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
class StaticTokenReplacer(nn.Module):
    def __init__(self, token_embeddings, pos_embed_layer):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = self.token_embeddings.shape[0]        
        self.pos_embed = pos_embed_layer        
        self.token_logits = nn.Parameter(torch.zeros(self.vocab_size))
        
    def forward(self, x, temperature, hard=False, tmp_position_idx=1):
        batch_size = x.shape[0]        
        soft_weights = F.gumbel_softmax(
            self.token_logits.unsqueeze(0).expand(batch_size, -1),
            tau=temperature,
            hard=hard
        )
        replacement_token_emb = torch.einsum("bv,vd->bd", soft_weights, self.token_embeddings)        
        pos_emb = self.pos_embed(torch.tensor(tmp_position_idx, device=x.device))  
        replacement_emb = replacement_token_emb + pos_emb.unsqueeze(0)  
        
        x_modified = x.clone()
        x_modified[:, tmp_position_idx, :] = replacement_emb
        
        return x_modified

class Embedding_Module(nn.Module):
    def __init__(self, pos_embed_data, token_embed_data, trainable=False):
        super(Embedding_Module, self).__init__()

        if not trainable:
            self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(pos_embed_data.copy()), dtype=torch.float32),freeze=True)
            self.token_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32),freeze=True)

            self.pos_embed.requires_grad_(False)
            self.token_embed.requires_grad_(False)
            
        else:
            self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(pos_embed_data.copy()), dtype=torch.float32),freeze=False)
            self.token_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32),freeze=False)
            
    def forward(self, x):
        token_embeddings = self.token_embed(x)
        position_embeddings = self.pos_embed(torch.arange(x.shape[-1], device=x.device))
        return (token_embeddings + position_embeddings).float()

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


import torch
import torch.nn as nn
# from AlgorithmicCounting1D.ModelComponent import mlp
# from AlgorithmicCounting1D.ModelComponent import attention

class Transformer_Module(nn.Module):
    def __init__(self, model_config, model_parameters, trainable=False, n_layers=1):
        super(Transformer_Module, self).__init__() 
        if not trainable:
            self.layers = []
            with torch.no_grad():
                for l in range(model_config.num_layers):
                    ATT_Layer = ATT_Module(key_size = model_config.key_size, 
                                                    num_heads=model_config.num_heads,
                                                    query_data=model_parameters['layers'][l]['attn']['query'],
                                                    key_data=model_parameters['layers'][l]['attn']['key'],
                                                    value_data=model_parameters['layers'][l]['attn']['value'],
                                                    linear_data=model_parameters['layers'][l]['attn']['linear'],
                                                    trainable=False)
                    ATT_Layer.requires_grad_(False)
                    self.layers.append(ATT_Layer)

                    MLP_Layer = MLP_Module(activation_function=model_config.activation_function, 
                                            linear_1_data=model_parameters['layers'][l]['mlp']['linear_1'],
                                            linear_2_data=model_parameters['layers'][l]['mlp']['linear_2'],
                                            trainable=False)
                    MLP_Layer.requires_grad_(False)
                    self.layers.append(MLP_Layer)

                self.model = nn.Sequential(*self.layers)

            self.model.requires_grad_(False)

        else:
            self.layers = []
            for _ in range(n_layers):
                ATT_Layer = ATT_Module(key_size = model_config.key_size, 
                                                num_heads=model_config.num_heads,
                                                query_data=model_parameters['layers'][0]['attn']['query'],
                                                key_data=model_parameters['layers'][0]['attn']['key'],
                                                value_data=model_parameters['layers'][0]['attn']['value'],
                                                linear_data=model_parameters['layers'][0]['attn']['linear'],
                                                trainable=True)
                self.layers.append(ATT_Layer)

                MLP_Layer = MLP_Module(activation_function=model_config.activation_function, 
                                        linear_1_data=model_parameters['layers'][0]['mlp']['linear_1'],
                                        linear_2_data=model_parameters['layers'][0]['mlp']['linear_2'],
                                        trainable=True)
                self.layers.append(MLP_Layer)

                self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


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
