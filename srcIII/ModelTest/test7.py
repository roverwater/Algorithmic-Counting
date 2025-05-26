import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA

# Create 224x224 toy image with large objects
def create_toy_image(size=224):
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    half = size // 2
    
    # Red square (top-left quadrant)
    draw.rectangle([(0, 0), (half-1, half-1)], fill=(255, 0, 0))
    
    # Blue square (top-right quadrant)
    draw.rectangle([(half, 0), (size-1, half-1)], fill=(0, 0, 255))
    
    # Red diamond (bottom-left)
    draw.polygon([(half//4, half), (half//2, half*3//2),
                 (half//4, half*2), (0, half*3//2)], fill=(255, 0, 0))
    
    # Blue diamond (bottom-right)
    draw.polygon([(half*3//4, half), (half, half*3//2),
                 (half*3//4, half*2), (half//2, half*3//2)], fill=(0, 0, 255))
    
    return img

# Load DINOv2 model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()

# Preprocessing
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Process image
img = create_toy_image()
tensor = transform(img).unsqueeze(0)

# Extract features
with torch.no_grad():
    features = model.forward_features(tensor)
    embeddings = features['x_norm_patchtokens'][0].cpu().numpy()

# Visualize with PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeddings)
pca_normalized = (pca_result - pca_result.min(0)) / (np.ptp(pca_result, 0)) * 255

# Create overlay image
patch_size = model.patch_size  # 14 for ViT-S/14
overlay = Image.new('RGB', img.size)
for i, color in enumerate(pca_normalized.astype(int)):
    y = (i // (img.width // patch_size)) * patch_size
    x = (i % (img.width // patch_size)) * patch_size
    patch = Image.new('RGB', (patch_size, patch_size), tuple(color))
    overlay.paste(patch, (x, y))

# Display results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(overlay)
ax[1].set_title('DINOv2 Embedding Similarity')
plt.show()