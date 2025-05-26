import torch                                                                                                                              
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from srcIII import config
from matplotlib.colors import ListedColormap
import torchvision.transforms as transforms

class DINOv2_Image_Transformer(nn.Module):
    def __init__(self, vocab_size=256):
        super().__init__()
        # Load DINOv2 and keep it trainable
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.train()  # Keep in train mode to allow gradients
        
        # number of patches per side (224 / 14 = 16)
        self.grid_size = 16
        
        # how many patches to take on each side
        self.n = 6
        
        # compute the start index so the n×n window is centered
        start = (self.grid_size - self.n) // 2
        end   = start + self.n
        
        # build flat indices into the 16×16 grid
        self.indices = [
            row*self.grid_size + col
            for row in range(start, end)
            for col in range(start, end)
        ]
        
        # Trainable projection to 1D per patch
        self.projection = nn.Sequential(
            nn.Linear(384, 128),  # Process each patch's 384-dim features
            nn.GELU(),
            nn.Linear(128, 1),     # Reduce to 1 dimension per patch
            nn.Tanh()
        )
        
        # Trainable discretization layer
        self.codebook = nn.Parameter(torch.linspace(-1, 1, vocab_size))
        
        # Preprocessing (now part of computation graph)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        x_list_str = []
        
        # Process each image in batch
        for img in x:
            # Convert to PIL and apply trainable transforms
            img_pil = transforms.ToPILImage()(img.cpu())
            img_tensor = self.transform(img_pil).unsqueeze(0).to(config.device)
            
            # Extract features WITH GRADIENTS
            features = self.model.forward_features(img_tensor)
            patch_features = features["x_norm_patchtokens"].squeeze(0)
            
            # Process middle 64 patches
            selected = patch_features[self.indices]  # [64, 384]
            
            # Project to 1D per patch
            projected = self.projection(selected)  # [64, 1]
            
            # Differentiable discretization
            quantized = torch.argmin(
                torch.abs(projected - self.codebook.unsqueeze(0)),
                dim=-1
            )  # [64]
            
            # Format tokens
            prefix = [config.bos, config.tmp, config.sep]
            tokens = quantized.cpu().numpy().flatten().tolist()

            # print(prefix + list(map(str, tokens)))

            x_list_str.append(prefix + list(map(str, tokens)))
            
        return x_list_str