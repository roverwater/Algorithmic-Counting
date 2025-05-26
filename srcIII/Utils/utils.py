import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import numpy as np

def extract_weights(data):
    structured_data = {'embeddings': {}}
    layers = {}

    for key, value in data.items():
        if key in ('pos_embed', 'token_embed'): 
            structured_data['embeddings'][key] = value
        else: 
            parts = key.split('/')
            layer_num = int(parts[1].split('_')[1])  
            module = parts[2]  
            submodule = parts[3]  

            if layer_num not in layers:
                layers[layer_num] = {'attn': {}, 'mlp': {}}
            
            layers[layer_num][module][submodule] = value

    structured_data['layers'] = layers
    return structured_data

class SquaresTrianglesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
        for f in self.file_list:
            parts = f.split('_')
            if len(parts) != 5 or not parts[0].isdigit():
                raise ValueError(f"Invalid filename format: {f}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')
        
        filename = self.file_list[idx]
        parts = filename.split('_')
        squares = int(parts[2])
        triangles = int(parts[4].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
            
        labels = torch.tensor([squares, triangles], dtype=torch.float32)
        return image, labels

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_dataloaders(root_dir, batch_size=32, split_ratios=(0.7, 0.15, 0.15), seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    full_dataset = SquaresTrianglesDataset(
        root_dir=root_dir,
        transform=get_transforms()
    )
    
    total = len(full_dataset)
    train_len = int(split_ratios[0] * total)
    val_len = int(split_ratios[1] * total)
    test_len = total - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# if __name__ == "__main__":
#     dataset_path = "shapes_sq0p40_tr0p30_sz1-3_off-2-2_pat64"
    
#     train_loader, val_loader, test_loader = get_dataloaders(
#         root_dir=dataset_path,
#         batch_size=32,
#         split_ratios=(0.7, 0.15, 0.15)
    
#     batch_images, batch_labels = next(iter(train_loader))
#     print(f"Batch image shape: {batch_images.shape}")
#     print(f"Batch labels shape: {batch_labels.shape}")
#     print(f"Sample labels: {batch_labels[0]}")