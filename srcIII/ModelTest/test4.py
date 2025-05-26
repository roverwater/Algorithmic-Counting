import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import models, transforms

# 1. Create a toy image with larger objects
def create_toy_image(size=448):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    obj_size = 64
    
    # Red square (top-left)
    draw.rectangle([(50, 50), (50+obj_size, 50+obj_size)], fill="red")
    # Blue diamond (bottom-right)
    diamond_points = [(300, 300), (300+obj_size, 300), (300+obj_size//2, 300+obj_size)]
    draw.polygon(diamond_points, fill="blue")
    return image

# 2. Load ResNet50 (without final layers)
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc
model.eval()

# 3. Preprocess image and extract feature map
preprocess = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = create_toy_image()
inputs = preprocess(image).unsqueeze(0)
with torch.no_grad():
    feature_map = model(inputs)  # Shape: (1, 2048, 14, 14)

# 4. Compute similarity to reference patch (red square at position 1,1)
ref_x, ref_y = 1, 1
reference = feature_map[:, :, ref_y, ref_x].unsqueeze(-1).unsqueeze(-1)
similarities = torch.nn.functional.cosine_similarity(feature_map, reference, dim=1)
similarities = similarities.squeeze().numpy()

# 5. Visualize heatmap
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(similarities, cmap="RdBu", extent=(0, image.width, image.height, 0))
plt.colorbar(label="Cosine Similarity")
plt.title("Similarity to Red Square (ResNet50)")
plt.show()