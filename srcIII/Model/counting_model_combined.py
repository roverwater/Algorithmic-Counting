import sys
sys.path.append('/home/ruov/projects/AlgorithmicCounting/')

import torch
import numpy as np
import torch.nn as nn
from srcIII import config
import torch.nn.functional as F
import torchvision.transforms as transforms

class Model_Image(nn.Module):
    def __init__(self, model_config, model_parameters, unembedding_data, encoding_func, encoded_vocab):
        super(Model_Image, self).__init__()

        vocab_size = model_parameters['embeddings']['token_embed']['embeddings'].shape[0] - 5

        self.image_transformer = DINOv2_Image_Transformer(
            vocab_size=vocab_size
        ).to(config.device)

        self.embedding_dim = unembedding_data.shape[0]

        self.embedder = Embedding_Module(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings'],
                                                                  trainable=False).to(config.device)

        self.classifier = Classifier(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings']).to(config.device)
        
        self.transformer = Transformer_Module(model_config=model_config,
                                                                         model_parameters=model_parameters,
                                                                         trainable=False,
                                                                         )
        
        self.unembedder = Unembedding_Module(unembedding_data=unembedding_data,
                                                                        use_unembed_argmax=False,
                                                                        trainable=False)
        self.embedder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.unembedder.requires_grad_(False)
            
    def forward(self, x, temperature=0.1):
        x = self.image_transformer(x)
        x = self.classifier(x, temperature=temperature)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
class DINOv2_Image_Transformer(nn.Module):
    def __init__(self, vocab_size=256):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False        
        self.grid_size = 16        
        self.n = 6        
        start = (self.grid_size - self.n) // 2
        end   = start + self.n        
        self.indices = [
            row*self.grid_size + col
            for row in range(start, end)
            for col in range(start, end)
        ]        
        self.projection = nn.Sequential(
            nn.Linear(384, 128),  # Process each patch's 384-dim features
            nn.GELU(),
            nn.Linear(128, 1),     # Reduce to 1 dimension per patch
            nn.Tanh()
        )
        
        self.codebook = nn.Parameter(torch.linspace(-1, 1, vocab_size))
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        # Preprocess images (batch-compatible)
        x = torch.stack([self.transform(transforms.ToPILImage()(img.cpu())) for img in x]).to(config.device)
        
        # Extract DINOv2 features
        with torch.no_grad():
            features = self.model.forward_features(x)
        patch_features = features["x_norm_patchtokens"][:, self.indices, :]  # [B, 64, 384]
        
        # Project to 1D
        projected = self.projection(patch_features).squeeze(-1)  # [B, 64]
        
        # STE for quantization
        distances = torch.abs(projected.unsqueeze(-1) - self.codebook)  # [B, 64, vocab_size]
        quantized_indices = torch.argmin(distances, dim=-1)  # Hard indices
        
        # STE: Use hard indices in forward, soft in backward
        soft_assign = F.softmax(-distances / 0.1, dim=-1)
        quantized_ste = (quantized_indices.float() - (soft_assign * distances).sum(-1).detach() + (soft_assign * distances).sum(-1))
        quantized_ste = quantized_ste.long()  # [B, 64]
        
        # Add special tokens
        prefix = torch.tensor([config.encoding_map[config.bos], config.encoding_map[config.tmp], config.encoding_map[config.sep]], device=x.device).expand(batch_size, 3)
        # prefix = torch.tensor([config.bos, config.tmp, config.sep], device=x.device).expand(batch_size, 3)
        tokens = torch.cat([prefix, quantized_ste], dim=1)  # [B, 67]
        return tokens

class Classifier(nn.Module):
    def __init__(self, pos_embed_data, token_embed_data):
        super(Classifier, self).__init__()

        self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(pos_embed_data.copy()), dtype=torch.float32),freeze=True)
        self.token_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32),freeze=True)
        self.token_embed_matrix = torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32)
        self.vocab_size = self.token_embed_matrix.shape[0]   
        self.token_logits = nn.Parameter(torch.zeros(self.vocab_size))
        self.pos_embed.requires_grad_(False)
        self.token_embed.requires_grad_(False)

    def forward(self, x, temperature):
        token_embeddings = self.token_embed(x)
        batch_size, _, _ = token_embeddings.shape
        
        soft_weights = F.gumbel_softmax(
            self.token_logits.unsqueeze(0).expand(batch_size, -1),
            tau=temperature,
            hard=False
        ) 

        learned_token_to_count = torch.einsum("bv,vd->bd", soft_weights, self.token_embed_matrix.to(x.device)).unsqueeze(1) 
        token_embeddings[:, 1, :] = learned_token_to_count.squeeze(1)       
        position_embeddings = self.pos_embed(torch.arange(x.shape[-1], device=x.device))
        return (token_embeddings + position_embeddings).float()

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
            pass

    def forward(self, x):
        x = self.model(x)
        return x

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





