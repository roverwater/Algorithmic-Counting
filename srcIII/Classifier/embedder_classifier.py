import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from srcIII import config

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

