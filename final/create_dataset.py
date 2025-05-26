import os
import torch
import numpy as np
import torch.nn as nn
from skimage.draw import polygon
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # or another interactive backend, such as Qt5Agg

# Image generation parameters
IMAGE_SIZE = 32
SAFE_MARGIN = 2
N_SAMPLES = 1000
N_LABELS = 7
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32

# Helper functions
def can_place_shape(image, top, left, height, width, gap=3):
    r_start = max(top - gap, 0)
    r_end = min(top + height + gap, image.shape[0])
    c_start = max(left - gap, 0)
    c_end = min(left + width + gap, image.shape[1])
    return np.all(image[r_start:r_end, c_start:c_end] == 0)

def can_place_polygon(image, r_coords, c_coords, gap=3):
    r_min = max(np.min(r_coords) - gap, 0)
    r_max = min(np.max(r_coords) + gap, image.shape[0])
    c_min = max(np.min(c_coords) - gap, 0)
    c_max = min(np.max(c_coords) + gap, image.shape[1])
    return np.all(image[r_min:r_max, c_min:c_max] == 0)

def generate_image(n_red_objects, num_distractors=3, num_extra_triangles=1, num_diamonds=1):
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # Place red objects
    placed = 0
    attempts = 0
    max_attempts = 300
    while placed < n_red_objects and attempts < max_attempts:
        shape_type = np.random.choice(["square", "triangle", "diamond"])

        if shape_type == "square":
            size = np.random.randint(2, 7)
            top = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - size - SAFE_MARGIN)
            left = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - size - SAFE_MARGIN)
            if can_place_shape(image, top, left, size, size):
                image[top:top + size, left:left + size] = 1
                placed += 1

        elif shape_type == "triangle":
            height = np.random.randint(4, 11)
            width = np.random.randint(4, 11)
            top = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - height - SAFE_MARGIN)
            left = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - width - SAFE_MARGIN)
            rr, cc = polygon([top, top + height, top], [left, left, left + width])
            if can_place_shape(image, top, left, height, width) and np.all(image[rr, cc] == 0):
                image[rr, cc] = 1
                placed += 1

        else:  # shape_type == "diamond"
            size = np.random.randint(2, 6)
            center_r = np.random.randint(SAFE_MARGIN + size, IMAGE_SIZE - size - SAFE_MARGIN)
            center_c = np.random.randint(SAFE_MARGIN + size, IMAGE_SIZE - size - SAFE_MARGIN)
            r = [center_r - size, center_r, center_r + size, center_r]
            c = [center_c, center_c + size, center_c, center_c - size]
            if can_place_polygon(image, r, c):
                rr, cc = polygon(r, c)
                if np.all(image[rr, cc] == 0):
                    image[rr, cc] = 1
                    placed += 1

        attempts += 1

    # Place distractors
    for _ in range(num_distractors):
        shape_type = np.random.choice(["square", "triangle", "diamond"])
        color = np.random.choice([2, 3])

        if shape_type == "square":
            size = np.random.randint(2, 7)
            top = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - size - SAFE_MARGIN)
            left = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - size - SAFE_MARGIN)
            if can_place_shape(image, top, left, size, size):
                image[top:top + size, left:left + size] = color

        elif shape_type == "triangle":
            height = np.random.randint(4, 11)
            width = np.random.randint(4, 11)
            top = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - height - SAFE_MARGIN)
            left = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - width - SAFE_MARGIN)

            orientation = np.random.choice(["up", "down", "left", "right"])
            if orientation == "up":
                r_coords = [top + height, top,         top        ]
                c_coords = [left,        left,         left + width]
            elif orientation == "down":
                r_coords = [top,         top + height, top + height]
                c_coords = [left,        left,         left + width]
            elif orientation == "left":
                r_coords = [top,         top + height, top        ]
                c_coords = [left + width, left + width, left       ]
            else:  # right
                r_coords = [top,         top + height, top + height]
                c_coords = [left,        left,         left + width]

            if can_place_polygon(image, r_coords, c_coords):
                rr, cc = polygon(r_coords, c_coords)
                if np.all(image[rr, cc] == 0):
                    image[rr, cc] = color

        else:  # shape_type == "diamond"
            size = np.random.randint(2, 6)
            center_r = np.random.randint(SAFE_MARGIN + size, IMAGE_SIZE - size - SAFE_MARGIN)
            center_c = np.random.randint(SAFE_MARGIN + size, IMAGE_SIZE - size - SAFE_MARGIN)
            r = [center_r - size, center_r, center_r + size, center_r]
            c = [center_c, center_c + size, center_c, center_c - size]
            if can_place_polygon(image, r, c):
                rr, cc = polygon(r, c)
                if np.all(image[rr, cc] == 0):
                    image[rr, cc] = color

    # Extra triangles and diamonds
    for _ in range(num_extra_triangles):
        base_r = np.array([21, 17, 25])
        base_c = np.array([20, 24, 20])
        r_min, r_max = base_r.min(), base_r.max()
        c_min, c_max = base_c.min(), base_c.max()
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        offset_r = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - height - SAFE_MARGIN)
        offset_c = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - width - SAFE_MARGIN)
        r = base_r - r_min + offset_r
        c = base_c - c_min + offset_c
        if can_place_polygon(image, r, c):
            rr, cc = polygon(r, c)
            image[rr, cc] = np.random.choice([2, 3])

    for _ in range(num_diamonds):
        size = np.random.randint(2, 6)
        center_r = np.random.randint(SAFE_MARGIN + size, IMAGE_SIZE - size - SAFE_MARGIN)
        center_c = np.random.randint(SAFE_MARGIN + size, IMAGE_SIZE - size - SAFE_MARGIN)
        r = [center_r - size, center_r, center_r + size, center_r]
        c = [center_c, center_c + size, center_c, center_c - size]
        if can_place_polygon(image, r, c):
            rr, cc = polygon(r, c)
            image[rr, cc] = np.random.choice([2, 3])

    return image, placed

def generate_and_save_dataset(save_dir, num_samples_per_label, labels):
    os.makedirs(save_dir, exist_ok=True)
    all_images = []
    all_labels = []
    for label in labels:
        for _ in range(num_samples_per_label):
            img, actual_label = generate_image(label)
            all_images.append(img)
            all_labels.append(actual_label)
    images_array = np.stack(all_images)
    labels_array = np.array(all_labels)
    np.save(os.path.join(save_dir, 'images.npy'), images_array)
    np.save(os.path.join(save_dir, 'labels.npy'), labels_array)

class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.images = np.load(os.path.join(data_dir, 'images.npy'))
        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        self.kernels = [
            torch.tensor([[[-1., 1.]]]).unsqueeze(0),
            torch.tensor([[[-1.], [1.]]]).unsqueeze(0),
            torch.tensor([[[0., 1.], [-1., 0.]]]).unsqueeze(0),
            torch.tensor([[[1., 0.], [0., -1.]]]).unsqueeze(0),
        ]
        self.pool = nn.MaxPool2d(kernel_size=6, stride=6)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        processed = self._process_image(img)
        return processed, torch.tensor(label, dtype=torch.long)

    def _process_image(self, img):
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        for kernel in self.kernels:
            conv = nn.Conv2d(1, 1, kernel.shape[-2:], padding='same', bias=False)
            with torch.no_grad():
                conv.weight.copy_(kernel)
            tensor = conv(tensor)
            tensor = torch.relu(tensor)
        tensor = self.pool(tensor)
        return tensor.squeeze()

def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs).unsqueeze(1), torch.stack(labels)

def GenerateDataset(data_dir='./saved_dataset'):
    # Generate data if not present
    if not os.path.exists(os.path.join(data_dir, 'images.npy')) or not os.path.exists(os.path.join(data_dir, 'labels.npy')):
        num_samples_per_label = N_SAMPLES // N_LABELS
        labels = range(N_LABELS)
        generate_and_save_dataset(data_dir, num_samples_per_label, labels)
    
    # Load dataset
    full_dataset = ImageDataset(data_dir)
    
    # Split dataset
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_loader, test_loader

def visualize_samples(m=5, data_dir='./saved_dataset'):
    # Generate/load dataset first
    GenerateDataset(data_dir)
    
    # Load the full dataset
    dataset = ImageDataset(data_dir)
    
    # Create colormap for original images
    colors = ['black', 'red', 'green', 'blue']
    cmap = ListedColormap(colors)
    
    # Randomly select m samples
    indices = np.random.choice(len(dataset), m, replace=False)
    
    # Create figure
    plt.figure(figsize=(10, 2*m))
    
    for i, idx in enumerate(indices):
        # Get original and processed images
        original = dataset.images[idx]
        processed = dataset[idx][0].detach().numpy()
        label = dataset.labels[idx]
        
        # Plot original image
        plt.subplot(m, 2, 2*i+1)
        plt.imshow(original, cmap=cmap, vmin=0, vmax=3)
        plt.title(f"Original (Label: {label})")
        plt.axis('off')
        
        # Plot processed image
        plt.subplot(m, 2, 2*i+2)
        # Normalize processed image for better visualization
        processed_norm = processed
        plt.imshow(processed_norm, cmap=cmap, vmin=0, vmax=3)
        plt.title("Processed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage example:
visualize_samples(m=6)