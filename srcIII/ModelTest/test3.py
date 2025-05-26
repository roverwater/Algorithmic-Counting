import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1. Create a toy image with red/blue squares and diamonds
def create_toy_image(size=224):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    
    # Red square (top-left)
    draw.rectangle([(50, 50), (100, 100)], fill="red")
    # Blue square (top-right)
    draw.rectangle([(150, 50), (200, 100)], fill="blue")
    # Red diamond (bottom-left)
    diamond_points = [(50, 150), (100, 150), (75, 200), (25, 200)]
    draw.polygon(diamond_points, fill="red")
    # Blue diamond (bottom-right)
    diamond_points = [(150, 150), (200, 150), (175, 200), (125, 200)]
    draw.polygon(diamond_points, fill="blue")
    return image

# 2. Load DINOv2 model and processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

# 3. Process image and extract patch embeddings
image = create_toy_image()
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # Shape: (1, num_patches + 1, 768)

# Remove [CLS] token and normalize embeddings
patch_embeddings = embeddings[:, 1:, :]
patch_embeddings = torch.nn.functional.normalize(patch_embeddings, dim=-1)

# 4. Select a reference patch (red square at center: 75,75)
ref_x, ref_y = 75, 75
patch_size = 14  # DINOv2 splits the image into 14x14 patches
ref_patch_x = ref_x // patch_size
ref_patch_y = ref_y // patch_size
ref_idx = ref_patch_y * (image.width // patch_size) + ref_patch_x
ref_embedding = patch_embeddings[0, ref_idx]

# 5. Compute cosine similarity for all patches
similarities = torch.matmul(patch_embeddings[0], ref_embedding).numpy()

# 6. Create a heatmap overlay
cmap = plt.get_cmap("RdBu")
norm_similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())

# Convert similarities to RGBA colors
overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
draw_overlay = ImageDraw.Draw(overlay)

for i in range(image.height // patch_size):
    for j in range(image.width // patch_size):
        idx = i * (image.width // patch_size) + j
        color = cmap(norm_similarities[idx])
        r, g, b, a = [int(c * 255) for c in color]
        a = 128  # 50% opacity
        x0, y0 = j * patch_size, i * patch_size
        x1, y1 = x0 + patch_size, y0 + patch_size
        draw_overlay.rectangle([x0, y0, x1, y1], fill=(r, g, b, a))

# 7. Overlay heatmap on the original image
image_with_overlay = Image.alpha_composite(image.convert("RGBA"), overlay)

# Display results
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(image_with_overlay)
ax[1].set_title("Similarity to Red Square (Center Patch)")
ax[1].axis("off")

plt.show()