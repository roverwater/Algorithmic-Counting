import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

# Create toy image with red/blue squares and diamonds
def create_toy_image(size=128):
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw red square (top-left quadrant)
    draw.rectangle([(0, 0), (63, 63)], fill='red')
    
    # Draw blue square (top-right quadrant)
    draw.rectangle([(64, 0), (127, 63)], fill='blue')
    
    # Draw red diamond (bottom-left quadrant)
    draw.polygon([(31, 64), (63, 96), (31, 127), (0, 96)], fill='red')
    
    # Draw blue diamond (bottom-right quadrant)
    draw.polygon([(95, 64), (127, 96), (95, 127), (64, 96)], fill='blue')
    
    return img

# Create dataset from image patches
class PatchDataset(Dataset):
    def __init__(self, img, patch_size=16):
        self.patches = []
        self.labels = []
        
        for y in range(0, img.height, patch_size):
            for x in range(0, img.width, patch_size):
                patch = img.crop((x, y, x+patch_size, y+patch_size))
                self.patches.append(np.array(patch) / 255.0)
                
                # Determine label based on quadrant
                if y < 64:
                    self.labels.append(0 if x < 64 else 1)  # Red/blue square
                else:
                    self.labels.append(2 if x < 64 else 3)   # Red/blue diamond

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].transpose(2, 0, 1)  # CHW format
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Simple CNN model
class PatchModel(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.flatten = nn.Flatten()
        self.embed = nn.Linear(32*4*4, embedding_dim)
        self.classify = nn.Linear(embedding_dim, 4)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))  # 16x16 -> 8x8
        x = self.pool(nn.functional.relu(self.conv2(x)))  # 8x8 -> 4x4
        x = self.flatten(x)
        embeddings = nn.functional.relu(self.embed(x))
        return self.classify(embeddings), embeddings

# Create image and dataset
img = create_toy_image()
dataset = PatchDataset(img)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model and training
model = PatchModel(embedding_dim=32)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(400):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = []
    for inputs, _ in dataloader:
        _, emb = model(inputs)
        embeddings.append(emb)
    embeddings = torch.cat(embeddings).numpy()

# Create PCA visualization
pca = PCA(n_components=3)
emb_pca = pca.fit_transform(embeddings)
emb_pca = (emb_pca - emb_pca.min(0)) / (emb_pca.max(0) - emb_pca.min(0)) * 255

# Create overlay image
overlay = Image.new('RGB', img.size)
for idx, (y, x) in enumerate([(y, x) for y in range(0, 128, 16) for x in range(0, 128, 16)]):
    color = tuple(emb_pca[idx].astype(int))
    patch = Image.new('RGB', (16, 16), color)
    overlay.paste(patch, (x, y))

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(overlay)
ax[1].set_title('Embedding Similarity Visualization')
plt.show()