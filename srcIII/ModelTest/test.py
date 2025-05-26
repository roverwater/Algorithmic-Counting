import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    patch_embeddings = features[:, 1:, :].squeeze(0).cpu().numpy()
    return patch_embeddings.reshape(14, 14, -1)

def visualize_dino_features(image_path):
    """Visualize DINO patch features and similarities"""
    # Load image
    orig_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = orig_image.shape[:2]
    
    # Get embeddings
    patch_emb = get_patch_embeddings(orig_image)
    
    # Create figure
    plt.figure(figsize=(20, 8))
    
    # 1. Original Image with Patch Grid
    plt.subplot(1, 3, 1)
    plt.imshow(orig_image)
    plt.title("Original Image with DINO Patches")
    
    # Draw patch grid
    patch_size = 16
    for x in np.linspace(0, w, 15):
        plt.axvline(x, color='white', linewidth=0.5, alpha=0.7)
    for y in np.linspace(0, h, 15):
        plt.axhline(y, color='white', linewidth=0.5, alpha=0.7)
    
    # 2. Patch Similarity Heatmap
    plt.subplot(1, 3, 2)
    flattened = patch_emb.reshape(196, -1)
    similarity = np.dot(flattened, flattened.T)
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar(label='Similarity Score')
    plt.title("Patch Similarity Matrix")
    
    # 3. 2D PCA Projection
    plt.subplot(1, 3, 3)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flattened)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=np.arange(196), 
                cmap='rainbow', edgecolor='k', s=50)
    plt.colorbar(label='Patch Index')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("2D PCA of Patch Embeddings")
    
    plt.tight_layout()
    plt.show()

def create_test_image():
    """Generate test image with colored squares"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Red squares
    img[50:100, 50:100] = [255, 0, 0]
    img[150:200, 150:200] = [255, 0, 0]
    # Blue squares
    img[50:100, 150:200] = [0, 0, 255]
    img[150:200, 50:100] = [0, 0, 255]
    return img

if __name__ == "__main__":
    import sys

    # image = create_test_image()
    # test_filepath = "test_image.png"
    # cv2.imwrite(test_filepath, image)


    test_filepath = "/home/ruov/projects/AlgorithmicCounting/data/custom_dataset/1000049724_6_aug_angle0_045.jpg"
    
    visualize_dino_features(test_filepath)