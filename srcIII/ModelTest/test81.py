from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def load_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    return model

def create_patches():
    patches = []
    for i in range(64):
        img = Image.new('RGB', (14, 14), color='black')
        draw = ImageDraw.Draw(img)
        size = (i//3) % 3 + 1
        h_offset = (i % 5) - 2
        v_offset = ((i//5) % 5) - 2
        
        if i % 2 == 0:  # Square
            base = (14 - size*3) // 2
            left = max(0, base + h_offset)
            top = max(0, base + v_offset)
            right = min(14, left + size*3)
            bottom = min(14, top + size*3)
            draw.rectangle([left, top, right, bottom], fill='red')
        elif i % 3 == 0:
            pass
        else:  # Triangle
            base_y = 7 + size*2 + v_offset
            apex_y = 7 - size*2 + v_offset
            points = [
                7 - size*2 + h_offset, base_y,
                7 + size*2 + h_offset, base_y,
                7 + h_offset, apex_y
            ]
            draw.polygon(points, fill='red')
        patches.append(img)
    return patches

def create_patch_grid(patches):
    full_image = Image.new('RGB', (14*8, 14*8))
    for i, patch in enumerate(patches):
        x, y = (i % 8)*14, (i // 8)*14
        full_image.paste(patch, (x, y))
    padded_image = Image.new('RGB', (224, 224), 'black')
    padded_image.paste(full_image, ((224-14*8)//2, (224-14*8)//2))
    return padded_image

def extract_features(model, image, indices):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    with torch.no_grad():
        features = model.forward_features(transform(image).unsqueeze(0))
        return features["x_norm_patchtokens"].squeeze().cpu().numpy()[indices]

def analyze_pca(embeddings, n_components=10):
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)
    return pca, embeddings_pca

def compute_similarities(embeddings, ref_idx):
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(normalized, normalized[ref_idx])

def plot_input_figure(image, patches):
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(image)
    ax1.set_title("Full Input Image", fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[1])
    gs_inner = gridspec.GridSpecFromSubplotSpec(8, 8, ax2.get_subplotspec(), 0.02, 0.02)
    for i, patch in enumerate(patches):
        ax = fig.add_subplot(gs_inner[i])
        ax.imshow(patch)
        ax.axis('off')
    ax2.set_title("Individual Patches Layout", fontsize=12)
    plt.tight_layout()

def plot_pca_figures(pca, embeddings_pca):
    # Variance plot including PC1
    fig1 = plt.figure(figsize=(12, 6))
    var_exp = pca.explained_variance_ratio_[:10] * 100
    plt.bar(range(1, 11), var_exp, color='teal', alpha=0.7)
    plt.title("Variance Explained by Principal Components (PC1-PC10)")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance (%)")
    plt.grid(alpha=0.3)
    plt.xticks(range(1, 11))
    
    # Component relationships including PC1
    fig2 = plt.figure(figsize=(18, 12))
    fig2.suptitle("PCA Component Relationships (Including PC1)", y=0.95)
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.4)
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (0,4), (1,4), (2,4)]
    
    for idx, (x, y) in enumerate(pairs[:6]):
        ax = fig2.add_subplot(gs[idx])
        for i, point in enumerate(embeddings_pca):
            color = 'red' if i%2==0 else 'blue'
            ax.scatter(point[x], point[y], color=color, s=30, alpha=0.6)
        ax.set_xlabel(f'PC{x+1}', fontsize=10)
        ax.set_ylabel(f'PC{y+1}', fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()

def plot_class_separation(f_values):
    fig = plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(f_values)+1), f_values, color='purple', alpha=0.7)
    plt.title("Class Separation Power per Principal Component (PC1-PC10)")
    plt.xlabel("Principal Component")
    plt.ylabel("F-value (ANOVA)")
    plt.grid(alpha=0.3)
    plt.xticks(range(1, len(f_values)+1))

def plot_similar_patches(patches, similarities, ref_idx):
    threshold = np.percentile(similarities, 60)
    similar_indices = np.where(similarities > threshold)[0]
    similar_indices = similar_indices[similar_indices != ref_idx]
    
    num_patches = len(similar_indices) + 1  # Include reference patch
    fig_width = max(6, num_patches * 1.2)  # Minimum width 6, 1.2" per patch
    
    fig = plt.figure(figsize=(fig_width, 3))
    plt.suptitle(f"Similar Patches to #{ref_idx+1} (Threshold: {threshold:.2f})", y=1.1)
    
    # Create flexible grid layout
    gs = gridspec.GridSpec(1, num_patches, width_ratios=[1.2]+[1]*(num_patches-1))
    
    # Reference patch
    ax0 = plt.subplot(gs[0])
    ax0.imshow(patches[ref_idx])
    ax0.set_title("Reference", fontsize=9)
    ax0.axis('off')
    
    # Similar patches
    for j, idx in enumerate(similar_indices):
        ax = plt.subplot(gs[j+1])
        ax.imshow(patches[idx])
        ax.set_title(f"#{idx+1}\n({similarities[idx]:.2f})", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()

def main():
    model = load_model()
    patches = create_patches()
    image = create_patch_grid(patches)
    
    indices = [row*16 + col for row in range(4,12) for col in range(4,12)]
    embeddings = extract_features(model, image, indices)
    
    pca, embeddings_pca = analyze_pca(embeddings)
    similarities = compute_similarities(embeddings_pca, 12)  # Now includes PC1
    f_values = f_classif(embeddings_pca, [0 if i%2==0 else 1 for i in range(64)])[0]
    
    plot_input_figure(image, patches)
    plot_pca_figures(pca, embeddings_pca)
    plot_class_separation(f_values)
    plot_similar_patches(patches, similarities, 12)
    plt.show()

 # Add new imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def mainII():
    model = load_model()
    patches = create_patches()
    image = create_patch_grid(patches)
    
    indices = [row*16 + col for row in range(4,12) for col in range(4,12)]
    embeddings = extract_features(model, image, indices)
    
    pca, embeddings_pca = analyze_pca(embeddings)
    similarities = compute_similarities(embeddings_pca, 12)
    f_values = f_classif(embeddings_pca, [0 if i%2==0 else 1 for i in range(64)])[0]
    
    # New code for classification and counting
    labels = np.array([0 if i % 2 == 0 else 1 for i in range(64)])
    
    # Train classifier on PCA features
    clf = LogisticRegression(max_iter=1000).fit(embeddings_pca, labels)
    predictions = clf.predict(embeddings_pca)
    square_count = (predictions == 0).sum()
    
    print(f"Predicted number of squares: {square_count}")
    
    # [Optional] Similarity-based counting
    threshold = np.median(similarities)  # Example threshold
    similarity_count = (similarities > threshold).sum()
    print(f"Similarity-based count: {similarity_count}")

    plot_input_figure(image, patches)
    plot_pca_figures(pca, embeddings_pca)
    plot_class_separation(f_values)
    plot_similar_patches(patches, similarities, 12)
    plt.show()   

if __name__ == "__main__":
    main()
    mainII()
    