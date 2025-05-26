import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt

def create_toy_image(size=448):
    """Create image with large red/blue shapes (squares and diamonds)"""
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    obj_size = 64  # Object size in pixels
    
    # Red square (top-left)
    draw.rectangle([(50, 50), (50+obj_size, 50+obj_size)], fill="red")
    # Blue square (top-right)
    draw.rectangle([(300, 50), (300+obj_size, 50+obj_size)], fill="blue")
    # Red diamond (bottom-left)
    diamond_points = [(50, 300), (50+obj_size, 300), 
                     (50+obj_size//2, 300+obj_size)]
    draw.polygon(diamond_points, fill="red")
    # Blue diamond (bottom-right)
    diamond_points = [(300, 300), (300+obj_size, 300),
                     (300+obj_size//2, 300+obj_size)]
    draw.polygon(diamond_points, fill="blue")
    return image

def main():
    # 1. Create image and setup model
    image = create_toy_image()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")

    # 2. Process image and get embeddings
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    # 3. Aggregate 2x2 patches to create larger regions
    patch_size = 14  # Original DINOv2 patch size (image_size=448 → 448/14=32 patches)
    aggregation = 2  # Combine 2x2 patches → 28x28 pixel regions
    
    # Reshape embeddings into grid (32x32 patches)
    h = w = image.width // patch_size
    emb_grid = embeddings[0].reshape(h, w, -1)  # [32, 32, 768]

    # Average pool embeddings over 2x2 regions
    emb_agg = torch.nn.functional.avg_pool2d(
        emb_grid.permute(2, 0, 1).unsqueeze(0),  # [1, 768, 32, 32]
        kernel_size=aggregation,
        stride=aggregation
    ).squeeze(0).permute(1, 2, 0)  # [16, 16, 768]

    # 4. Compute similarity to reference (red square at top-left)
    ref_x, ref_y = 1, 1  # Aggregated patch index covering red square
    ref_embedding = emb_agg[ref_y, ref_x]
    similarities = torch.matmul(emb_agg, ref_embedding).numpy()

    # 5. Create heatmap overlay
    cmap = plt.get_cmap("RdBu")
    norm_similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min())
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    agg_patch_size = patch_size * aggregation
    for i in range(emb_agg.shape[0]):
        for j in range(emb_agg.shape[1]):
            color = cmap(norm_similarities[i, j])
            r, g, b, a = [int(c * 255) for c in color]
            a = 128  # 50% opacity
            x0 = j * agg_patch_size
            y0 = i * agg_patch_size
            x1 = x0 + agg_patch_size
            y1 = y0 + agg_patch_size
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, a))

    # 6. Combine and display
    image_with_overlay = Image.alpha_composite(image.convert("RGBA"), overlay)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(image_with_overlay)
    ax[1].set_title("Similarity to Red Square (Aggregated DINOv2 Patches)")
    ax[1].axis("off")

    plt.show()

if __name__ == "__main__":
    main()