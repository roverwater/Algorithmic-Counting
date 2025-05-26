#!/usr/bin/env python3
"""
heatmap_vit_similarity.py

Compute and plot ViT patch‐based similarity maps as standalone heatmaps
for a toy image, without any RGBA overlays.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt


def create_toy_image(size: int = 224) -> Image.Image:
    """
    Create a toy RGB image with two colored squares and two colored diamonds.
    """
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    # red square
    draw.rectangle([(50, 50), (100, 100)], fill="red")
    # blue square
    draw.rectangle([(150, 50), (200, 100)], fill="blue")
    # red diamond
    draw.polygon([(50, 150), (100, 150), (75, 200), (25, 200)], fill="red")
    # blue diamond
    draw.polygon([(150, 150), (200, 150), (175, 200), (125, 200)], fill="blue")
    return image


def get_patch_embeddings(image: Image.Image,
                         processor: AutoImageProcessor,
                         model: AutoModel,
                         device: torch.device):
    """
    Run the image through the ViT model and return normalized patch embeddings.
    Returns:
        patch_emb: Tensor of shape (n_patches, dim)
        grid_w: number of patches along width
        grid_h: number of patches along height
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # skip class token at index 0
    patch_emb = outputs.last_hidden_state[:, 1:, :].squeeze(0)
    patch_emb = torch.nn.functional.normalize(patch_emb, dim=-1).cpu()
    n_tokens = patch_emb.shape[0]
    side = int(np.sqrt(n_tokens))
    return patch_emb, side, side


def cosine_similarity_map(patch_emb: torch.Tensor,
                          ref_emb: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarities between every patch embedding and a reference embedding.
    Returns a flat numpy array of length n_patches.
    """
    sims = torch.matmul(patch_emb, ref_emb).numpy()
    return sims


def bilinear_smooth(sim_values: np.ndarray,
                    grid_w: int, grid_h: int,
                    target_size: tuple[int, int]) -> np.ndarray:
    """
    Upsample the flat similarity map (grid_h x grid_w) to the full image resolution
    via bilinear interpolation.
    """
    sim_tensor = torch.tensor(sim_values, dtype=torch.float32).reshape(1, 1, grid_h, grid_w)
    up = torch.nn.functional.interpolate(sim_tensor,
                                         size=target_size[::-1],
                                         mode="bilinear", align_corners=False)
    return up.squeeze().numpy()


def main():
    # 1. Prepare image and model
    image = create_toy_image()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    # 2. Single‑patch reference (center of red square at pixel ~ (75,75))
    patch_emb, gw, gh = get_patch_embeddings(image, processor, model, device)
    ref_idx = (75 // 14) + (75 // 14) * gw
    ref_emb = patch_emb[ref_idx]
    orig_sims = cosine_similarity_map(patch_emb, ref_emb).reshape(gh, gw)

    # 3. Pooled‐object reference over the red square bbox
    xmin_p, ymin_p = 50 // 14, 50 // 14
    xmax_p, ymax_p = 100 // 14, 100 // 14
    patch_ids = [
        y * gw + x
        for y in range(ymin_p, ymax_p + 1)
        for x in range(xmin_p, xmax_p + 1)
    ]
    pooled_ref = torch.nn.functional.normalize(patch_emb[patch_ids].mean(0), dim=0)
    pooled_sims = cosine_similarity_map(patch_emb, pooled_ref).reshape(gh, gw)

    # 4. Coarse‐scale input (112×112 → 8×8 patches)
    coarse_img = image.resize((112, 112), Image.BILINEAR)
    with torch.no_grad():
        coarse_out = model(**processor(images=coarse_img,
                                        do_resize=False,
                                        return_tensors="pt").to(device))
    coarse_emb = torch.nn.functional.normalize(
        coarse_out.last_hidden_state[:, 1:, :].squeeze(0), dim=-1
    ).cpu()
    cW = cH = int(np.sqrt(coarse_emb.shape[0]))
    ref_idx_c = (75 // 28) + (75 // 28) * cW
    coarse_sims = cosine_similarity_map(coarse_emb, coarse_emb[ref_idx_c]).reshape(cH, cW)

    # 5. Smoothed original similarity map at full resolution
    flat_orig = cosine_similarity_map(patch_emb, ref_emb)
    smoothed = bilinear_smooth(flat_orig, gw, gh, image.size)

    # 6. Plot the four heatmaps
    titles = [
        "(1) Single‑patch ref",
        "(2) Pooled ref",
        "(3) Coarse input",
        "(4) Smoothed map"
    ]
    maps = [orig_sims, pooled_sims, coarse_sims, smoothed]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, hm, title in zip(axes, maps, titles):
        # nearest interpolation for the small grids, bilinear for full-res
        interp = 'nearest' if hm.shape == orig_sims.shape else 'bilinear'
        ax.imshow(hm, cmap='RdBu', interpolation=interp)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
