import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
from sklearn.decomposition import PCA

# Load DINO model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').eval()

# Preprocessing transform
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
])

def get_patch_embeddings(image):
    """Extract DINO patch embeddings with proper padding"""
    h, w = image.shape[:2]
    scale = 224 / max(h, w)
    new_h, new_w = int(h*scale), int(w*scale)
    
    # Resize and pad to 224x224
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.zeros((224, 224, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Convert to tensor
    tensor = preprocess(padded).unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        features = model.get_intermediate_layers(tensor, n=1)[0]
    
    # Remove CLS token and reshape
    return features[:, 1:, :].squeeze(0).cpu().numpy().reshape(14, 14, -1)

def create_test_image():
    """Generate test image with colored squares"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[50:100, 50:100] = [255, 0, 0]    # Red
    img[150:200, 150:200] = [255, 0, 0]  # Red
    img[50:100, 150:200] = [0, 0, 255]   # Blue
    img[150:200, 50:100] = [0, 0, 255]   # Blue
    return img

def visualize_pca_overlay(image):
    """Main visualization function with PCA overlay"""
    h, w = image.shape[:2]
    patch_emb = get_patch_embeddings(image)
    
    # Compute PCA
    pca = PCA(n_components=3)
    flattened = patch_emb.reshape(-1, 384)
    pca_result = pca.fit_transform(flattened)
    
    # Normalize PCA components
    pca_min = pca_result.min(axis=0)
    pca_max = pca_result.max(axis=0)
    pca_normalized = (pca_result - pca_min) / (pca_max - pca_min)
    
    # Create color mask
    color_mask = np.zeros((h, w, 3))
    scale_x, scale_y = w/224, h/224
    
    for i in range(14):
        for j in range(14):
            y_start = int(i * 16 * scale_y)
            y_end = int((i+1) * 16 * scale_y)
            x_start = int(j * 16 * scale_x)
            x_end = int((j+1) * 16 * scale_x)
            color_mask[y_start:y_end, x_start:x_end] = pca_normalized[i*14 + j]

    # Visualization
    fig = plt.figure(figsize=(18, 8))
    
    # PCA Overlay
    ax1 = fig.add_subplot(121)
    ax1.imshow(image)
    ax1.imshow(color_mask, alpha=0.6)
    for x in np.linspace(0, w, 15):
        ax1.axvline(x, color='white', linewidth=0.3, alpha=0.5)
    for y in np.linspace(0, h, 15):
        ax1.axhline(y, color='white', linewidth=0.3, alpha=0.5)
    ax1.set_title("PCA Color Overlay")
    
    # 3D Scatter Plot (Fixed)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        pca_result[:,0], 
        pca_result[:,1], 
        pca_result[:,2],
        c=pca_normalized,
        s=50,  # Marker size parameter
        edgecolor='k'
    )
    ax2.set_title("3D PCA Projection")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    # image = create_test_image()
    # test_filepath = "test_image.png"
    # cv2.imwrite(test_filepath, image)


    test_filepath = "/home/ruov/projects/AlgorithmicCounting/data/custom_dataset/1000049724_6_aug_angle0_045.jpg"
    image = cv2.cvtColor(cv2.imread(test_filepath), cv2.COLOR_BGR2RGB)
    visualize_pca_overlay(image)