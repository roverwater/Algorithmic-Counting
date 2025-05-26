from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.gridspec as gridspec

# Load DINOv2 model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

# Create individual patches
patches = []
for i in range(16):
    if i % 2 == 0:  # Square patches
        img = Image.new('RGB', (14, 14), color='black')
        draw = ImageDraw.Draw(img)
        size = i%3
        draw.rectangle([(14-size)//2, (14-size)//2, (14+size)//2, (14+size)//2], fill='red')
    else:  # Triangle patches
        img = Image.new('RGB', (14, 14), color='black')
        draw = ImageDraw.Draw(img)
        offset = i%3
        draw.polygon([(offset, offset), (14-offset, offset), (7, 14-offset)], fill='red')
    patches.append(img)

# Create full image and pad for DINOv2
full_image = Image.new('RGB', (14*16, 14))
for i, patch in enumerate(patches):
    full_image.paste(patch, (i*14, 0))
padded_image = Image.new('RGB', (224, 224), color='black')
padded_image.paste(full_image, (0, 0))

# Transform and process
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
inputs = transform(padded_image).unsqueeze(0)

# Get embeddings
with torch.no_grad():
    features = model.forward_features(inputs)
    patch_embeddings = features["x_norm_patchtokens"].squeeze().cpu().numpy()[:16]

# Visualization with proper subplot arrangement
fig = plt.figure(figsize=(18, 6))

# Main grid: 1 row, 3 columns
gs_main = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 2, 1])

# Input image
ax1 = fig.add_subplot(gs_main[0])
ax1.imshow(padded_image)
ax1.set_title("Model Input Image")
ax1.axis('off')

# Individual patches grid
ax2 = fig.add_subplot(gs_main[1])
gs_inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs_main[1],
                                          wspace=0.05, hspace=0.05)
for i in range(16):
    ax = fig.add_subplot(gs_inner[i])
    ax.imshow(patches[i])
    ax.axis('off')
ax2.set_title("Individual Patches", y=1.05)

# PCA plot
ax3 = fig.add_subplot(gs_main[2])
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(patch_embeddings)
colors = ['red' if i%2==0 else 'blue' for i in range(16)]
for i, (x, y) in enumerate(embeddings_2d):
    ax3.scatter(x, y, color=colors[i], s=100)
    ax3.text(x, y, str(i+1), ha='center', va='bottom')
ax3.set_xlabel('PCA Component 1')
ax3.set_ylabel('PCA Component 2')
ax3.set_title('Patch Embeddings\n(Red=Squares, Blue=Triangles)')
ax3.grid(True)

plt.tight_layout()
plt.show()