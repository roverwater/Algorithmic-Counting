import torch
import numpy as np
import torch.nn as nn

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