import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import jax
import math
import torch
from torch import nn, optim
from matplotlib.colors import ListedColormap
from tracr.rasp import rasp
from tracr.compiler import compiling
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
import numpy as np
import torch.nn as nn
from skimage.draw import polygon
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class config():
    def __init__(self):
        self.debug = True
        self.bos = "BOS"
        self.tmp = "TMP" 
        self.sep = "SEP"
        self.max_rasp_len = 40
        self.mlp_exactness = 1000000
        self.vocab_req = {self.tmp, self.sep}
        self.vocab_tokens = {0, 1, 2, 3}
        self.vocab = self.vocab_tokens.union(self.vocab_req)
        self.input = [[self.bos, 3, self.sep, 3, 3, 0, 2, 1, 3]]

        self.model = None
        self.encoding_map = None
        self.encoded_bos = None
        self.out = None

        self.logit_dataset = True
        self.image_size = 32
        self.safe_margin = 2
        self.n_samples = 1000
        self.n_labels = 7
        self.train_split = 0.8
        self.batch_size = 512
        # self.batch_size = 2000
        # self.max_seq_len = 9
        # self.train_split = 0.9


        self.learning_rate = 1e-2
        self.num_epochs = 200
        self.temperature = 0.1

        self.start_temp = 0.3
        self.end_temp = 0.01

        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.rasp_model = None
        self.out_rasp = None
        self.model_config = None
        self.model_params = None
        self.embedding_dim = None
        self.output_dim = None
        self.unembed_matrix = None

def extract_weights(data):
    structured_data = {'embeddings': {}}
    layers = {}

    for key, value in data.items():
        if key in ('pos_embed', 'token_embed'): 
            structured_data['embeddings'][key] = value
        else: 
            parts = key.split('/')
            layer_num = int(parts[1].split('_')[1])  
            module = parts[2]  
            submodule = parts[3]  

            if layer_num not in layers:
                layers[layer_num] = {'attn': {}, 'mlp': {}}
            
            layers[layer_num][module][submodule] = value

    structured_data['layers'] = layers
    return structured_data

def count_agnostic_first():
        SELECT_ALL_TRUE = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        LENGTH = rasp.SelectorWidth(SELECT_ALL_TRUE) * 0
        SELECT_FIRST = rasp.Select(rasp.indices, LENGTH , rasp.Comparison.EQ)
        FIRST_TOKEN = rasp.Aggregate(SELECT_FIRST, rasp.tokens)
        COUNT = rasp.SelectorWidth(rasp.Select(rasp.tokens, FIRST_TOKEN, rasp.Comparison.EQ))
        return COUNT

def compile_model():
    model = compiling.compile_rasp_to_model(
        count_agnostic_first(),
        vocab=config.vocab,
        max_seq_len=config.max_rasp_len,
        compiler_bos=config.bos,
        mlp_exactness=config.mlp_exactness
    )
    return model

def compile():
    config.model = compile_model()
    config.encoding_map = config.model.input_encoder.encoding_map
    config.encoded_bos = config.encoding_map[config.bos]
    config.out = config.model.apply(config.input[0])
    config.unembed_matrix = config.out.unembed_matrix
    config.output_dim = config.unembed_matrix.shape[1]
    config.model_params = extract_weights(config.model.params)
    config.model_config = config.model.model_config
    config.model_config.activation_function = torch.nn.ReLU()

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
    image = np.zeros((config.image_size, config.image_size), dtype=np.float32)

    # Place red objects
    placed = 0
    attempts = 0
    max_attempts = 300
    while placed < n_red_objects and attempts < max_attempts:
        shape_type = np.random.choice(["square", "triangle", "diamond"])

        if shape_type == "square":
            size = np.random.randint(2, 7)
            top = np.random.randint(config.safe_margin, config.image_size - size - config.safe_margin)
            left = np.random.randint(config.safe_margin, config.image_size - size - config.safe_margin)
            if can_place_shape(image, top, left, size, size):
                image[top:top + size, left:left + size] = 1
                placed += 1

        elif shape_type == "triangle":
            height = np.random.randint(4, 11)
            width = np.random.randint(4, 11)
            top = np.random.randint(config.safe_margin, config.image_size - height - config.safe_margin)
            left = np.random.randint(config.safe_margin, config.image_size - width - config.safe_margin)
            rr, cc = polygon([top, top + height, top], [left, left, left + width])
            if can_place_shape(image, top, left, height, width) and np.all(image[rr, cc] == 0):
                image[rr, cc] = 1
                placed += 1

        else:  # shape_type == "diamond"
            size = np.random.randint(2, 6)
            center_r = np.random.randint(config.safe_margin + size, config.image_size - size - config.safe_margin)
            center_c = np.random.randint(config.safe_margin + size, config.image_size - size - config.safe_margin)
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
            top = np.random.randint(config.safe_margin, config.image_size - size - config.safe_margin)
            left = np.random.randint(config.safe_margin, config.image_size - size - config.safe_margin)
            if can_place_shape(image, top, left, size, size):
                image[top:top + size, left:left + size] = color

        elif shape_type == "triangle":
            height = np.random.randint(4, 11)
            width = np.random.randint(4, 11)
            top = np.random.randint(config.safe_margin, config.image_size - height - config.safe_margin)
            left = np.random.randint(config.safe_margin, config.image_size - width - config.safe_margin)

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
            center_r = np.random.randint(config.safe_margin + size, config.image_size - size - config.safe_margin)
            center_c = np.random.randint(config.safe_margin + size, config.image_size - size - config.safe_margin)
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
        offset_r = np.random.randint(config.safe_margin, config.image_size - height - config.safe_margin)
        offset_c = np.random.randint(config.safe_margin, config.image_size - width - config.safe_margin)
        r = base_r - r_min + offset_r
        c = base_c - c_min + offset_c
        if can_place_polygon(image, r, c):
            rr, cc = polygon(r, c)
            image[rr, cc] = np.random.choice([2, 3])

    for _ in range(num_diamonds):
        size = np.random.randint(2, 6)
        center_r = np.random.randint(config.safe_margin + size, config.image_size - size - config.safe_margin)
        center_c = np.random.randint(config.safe_margin + size, config.image_size - size - config.safe_margin)
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
        return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.long)

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
        num_samples_per_label = config.n_samples // config.n_labels
        labels = range(config.n_labels)
        generate_and_save_dataset(data_dir, num_samples_per_label, labels)
    
    # Load dataset
    full_dataset = ImageDataset(data_dir)
    
    # Split dataset
    train_size = int(config.train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create dataloaders
    config.train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    config.test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

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
        processed = dataset._process_image(original).detach().numpy()
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

class Model(nn.Module):
    def __init__(self, model_config, model_parameters, unembedding_data, encoding_map):
        super(Model, self).__init__()

        self.image_transformer = Image_Transformer()

        self.embedding_dim = unembedding_data.shape[0]

        self.encoder = encoding_map

        self.classifier = Classifier(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings']).to(config.device)
        
        self.transformer = Transformer_Module(model_config=model_config,
                                                                         model_parameters=model_parameters,
                                                                         trainable=False,
                                                                         )
        
        self.unembedder = Unembedding_Module(unembedding_data=unembedding_data,
                                                                        use_unembed_argmax=False,
                                                                        trainable=False)
        self.transformer.requires_grad_(False)
        self.unembedder.requires_grad_(False)
            
    def forward(self, x, temperature=0.1):
        x = self.image_transformer(x)
        x = x.long()
        x = self.classifier(x, temperature=temperature)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
class Image_Transformer(nn.Module):
    def __init__(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        super(Image_Transformer, self).__init__()

        self.kernels = [
            torch.tensor([[[-1., 1.]]]).unsqueeze(0),
            torch.tensor([[[-1.], [1.]]]).unsqueeze(0),
            torch.tensor([[[0., 1.], [-1., 0.]]]).unsqueeze(0),
            torch.tensor([[[1., 0.], [0., -1.]]]).unsqueeze(0),
        ]

        self.layers = []
        with torch.no_grad():
            for kernel in self.kernels:
                conv_layer = nn.Conv2d(1, 1, kernel.shape[-2:], padding='same', bias=False)
                with torch.no_grad():
                    conv_layer.weight.copy_(kernel)
                conv_layer.requires_grad_(False)
                self.layers.append(conv_layer)
                self.layers.append(nn.ReLU())

            self.conv_block = nn.Sequential(*self.layers)

        self.conv_block.requires_grad_(False)
        self.pool_block = nn.MaxPool2d(kernel_size=6, stride=6)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool_block(x)

        x_reshaped = x.view(x.size(0), -1)          

        prefix = torch.tensor([
            config.encoding_map[config.bos],
            config.encoding_map[config.tmp],
            config.encoding_map[config.sep],
        ], device=x_reshaped.device, dtype=x_reshaped.dtype)   # shape: (3,)

        prefix_batch = prefix.unsqueeze(0)            # shape: (1, 3)
        prefix_batch = prefix_batch.expand(x_reshaped.size(0), -1)  

        out = torch.cat([prefix_batch, x_reshaped], dim=1)  
        return out

class Classifier(nn.Module):
    def __init__(self, pos_embed_data, token_embed_data):
        super(Classifier, self).__init__()

        self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(pos_embed_data.copy()), dtype=torch.float32),freeze=True)
        self.token_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32),freeze=True)
        self.token_embed_matrix = torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32)
        self.vocab_size = self.token_embed_matrix.shape[0]   
        self.token_logits = nn.Parameter(torch.zeros(self.vocab_size))
        self.pos_embed.requires_grad_(False)
        self.token_embed.requires_grad_(False)

    def forward(self, x, temperature):
        token_embeddings = self.token_embed(x)
        batch_size, _, _ = token_embeddings.shape
        
        soft_weights = F.gumbel_softmax(
            self.token_logits.unsqueeze(0).expand(batch_size, -1),
            tau=temperature,
            hard=False
        ) 

        learned_token_to_count = torch.einsum("bv,vd->bd", soft_weights, self.token_embed_matrix.to(x.device)).unsqueeze(1) 
        token_embeddings[:, 1, :] = learned_token_to_count.squeeze(1)       
        position_embeddings = self.pos_embed(torch.arange(x.shape[-1], device=x.device))
        return (token_embeddings + position_embeddings).float()

class Embedding_Module(nn.Module):
    def __init__(self, pos_embed_data, token_embed_data, trainable=False):
        super(Embedding_Module, self).__init__()

        if not trainable:
            self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(pos_embed_data.copy()), dtype=torch.float32),freeze=True)
            self.token_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32),freeze=True)

            self.pos_embed.requires_grad_(False)
            self.token_embed.requires_grad_(False)
            
        else:
            self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(pos_embed_data.copy()), dtype=torch.float32),freeze=False)
            self.token_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32),freeze=False)
            
    def forward(self, x):
        token_embeddings = self.token_embed(x)
        position_embeddings = self.pos_embed(torch.arange(x.shape[-1], device=x.device))
        return (token_embeddings + position_embeddings).float()

class Transformer_Module(nn.Module):
    def __init__(self, model_config, model_parameters, trainable=False, n_layers=1):
        super(Transformer_Module, self).__init__() 
        if not trainable:
            self.layers = []
            with torch.no_grad():
                for l in range(model_config.num_layers):
                    ATT_Layer = ATT_Module(key_size = model_config.key_size, 
                                                    num_heads=model_config.num_heads,
                                                    query_data=model_parameters['layers'][l]['attn']['query'],
                                                    key_data=model_parameters['layers'][l]['attn']['key'],
                                                    value_data=model_parameters['layers'][l]['attn']['value'],
                                                    linear_data=model_parameters['layers'][l]['attn']['linear'],
                                                    trainable=False)
                    ATT_Layer.requires_grad_(False)
                    self.layers.append(ATT_Layer)

                    MLP_Layer = MLP_Module(activation_function=model_config.activation_function, 
                                            linear_1_data=model_parameters['layers'][l]['mlp']['linear_1'],
                                            linear_2_data=model_parameters['layers'][l]['mlp']['linear_2'],
                                            trainable=False)
                    MLP_Layer.requires_grad_(False)
                    self.layers.append(MLP_Layer)

                self.model = nn.Sequential(*self.layers)

            self.model.requires_grad_(False)

        else:
            pass

    def forward(self, x):
        x = self.model(x)
        return x

class ATT_Module(nn.Module):
    def __init__(self, key_size, num_heads, query_data, key_data, value_data, linear_data, trainable=False):
        super(ATT_Module, self).__init__()

        self.key_size = key_size
        self.num_heads = num_heads

        self.softmax = torch.nn.Softmax(dim=-1)  

        self.Query_Matrix = nn.Linear(query_data['w'].shape[0], query_data['w'].shape[1]) #In features/out features vs out features/in features for the weight matrix
        self.Key_Matrix = nn.Linear(key_data['w'].shape[0], key_data['w'].shape[1])
        self.Value_Matrix = nn.Linear(value_data['w'].shape[0], value_data['w'].shape[1])
        self.Linear_Matrix = nn.Linear(linear_data['w'].shape[0], linear_data['w'].shape[1])

        if not trainable:
            with torch.no_grad():
                self.Query_Matrix.bias.data = torch.tensor(np.array(query_data['b']).copy(), dtype=torch.float32)
                self.Query_Matrix.weight.data = torch.tensor(np.array(query_data['w']).copy(), dtype=torch.float32).T

                self.Key_Matrix.bias.data = torch.tensor(np.array(key_data['b']).copy(), dtype=torch.float32)
                self.Key_Matrix.weight.data = torch.tensor(np.array(key_data['w']).copy(), dtype=torch.float32).T

                self.Value_Matrix.bias.data = torch.tensor(np.array(value_data['b']).copy(), dtype=torch.float32)
                self.Value_Matrix.weight.data = torch.tensor(np.array(value_data['w']).copy(), dtype=torch.float32).T

                self.Linear_Matrix.bias.data = torch.tensor(np.array(linear_data['b']).copy(), dtype=torch.float32)
                self.Linear_Matrix.weight.data = torch.tensor(np.array(linear_data['w']).copy(), dtype=torch.float32).T

            self.Query_Matrix.requires_grad_(False)
            self.Key_Matrix.requires_grad_(False)
            self.Value_Matrix.requires_grad_(False)
            self.Linear_Matrix.requires_grad_(False)
            self.softmax.requires_grad_(False)            
    
    def forward(self, x):
        *leading_dims, _ = x.shape

        query = self.Query_Matrix(x)
        query = query.reshape((*leading_dims, self.num_heads, self.key_size))

        key = self.Key_Matrix(x)
        key = key.reshape((*leading_dims, self.num_heads, self.key_size))

        value = self.Value_Matrix(x)
        value = value.reshape((*leading_dims, self.num_heads, self.key_size))

        attn_logits = torch.einsum("...thd,...Thd->...htT", query, key)
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(np.float32)

        attn_weights = self.softmax(attn_logits) 
        attn = torch.einsum("...htT,...Thd->...thd", attn_weights, value)
        attn = torch.reshape(attn, (*leading_dims, -1))
        
        out = self.Linear_Matrix(attn)
        return x + out

class MLP_Module(nn.Module):
    def __init__(self, activation_function, linear_1_data, linear_2_data, trainable=False):
        super(MLP_Module, self).__init__()

        self.linear_1 = nn.Linear(linear_1_data['w'].shape[0], linear_1_data['w'].shape[1])
        self.activation_func = activation_function
        self.linear_2 = nn.Linear(linear_2_data['w'].shape[0], linear_2_data['w'].shape[1])  

        if not trainable:  
            with torch.no_grad():
                self.linear_1.bias.data = torch.tensor(np.array(linear_1_data['b']).copy(), dtype=torch.float32)
                self.linear_1.weight.data = torch.tensor(np.array(linear_1_data['w']).copy(), dtype=torch.float32).T

                self.linear_2.bias.data = torch.tensor(np.array(linear_2_data['b']).copy(), dtype=torch.float32)
                self.linear_2.weight.data = torch.tensor(np.array(linear_2_data['w']).copy(), dtype=torch.float32).T

            self.linear_1.requires_grad_(False)
            self.linear_2.requires_grad_(False)
            
    def forward(self, x):
        y = self.linear_1(x)
        y = self.activation_func(y)
        y = self.linear_2(y)
        return x + y

class Unembedding_Module(nn.Module):
    def __init__(self, unembedding_data, use_unembed_argmax, trainable=False):
        super(Unembedding_Module, self).__init__()

        self.use_unembed_argmax = use_unembed_argmax
        self.unembed = nn.Linear(unembedding_data.shape[0], unembedding_data.shape[1], bias=False)

        if not trainable:
            with torch.no_grad():
                self.unembed.weight.data = torch.tensor(np.array(unembedding_data).copy(), dtype=torch.float32).T

            self.unembed.requires_grad_(False)
            
    def forward(self, x):
        x = self.unembed(x)
        if(self.use_unembed_argmax):
            x = torch.argmax(x, axis=-1)
        return x

# Set device
jax.config.update("jax_default_matmul_precision", "highest")
matplotlib.use('TkAgg')  
config = config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {config.device}")

# Compile RASP and compute dataset
compile()
print(config.out.decoded) #prints: ['BOS', 4, 4, 4, 4, 4, 4, 4, 4]
# visualize_samples(m=6)
GenerateDataset()

# Move model to GPU
model = Model(model_config=config.model_config,
                          model_parameters=config.model_params,
                          unembedding_data=config.unembed_matrix,
                          encoding_map=config.encoding_map,
                          ).to(config.device)

#Visualize first batch
test_input, test_labels = next(iter(config.test_loader))
output_first_batch = model(test_input.to(config.device))
outputs_before_training = output_first_batch.argmax(dim=-1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

def log_token_preferences(model, epoch, history):
    token_probs = F.softmax(model.classifier.token_logits, dim=0).detach().cpu().numpy()
    history[epoch] = token_probs

def plot_token_preferences(history):
    epochs = sorted(history.keys())

    probs_over_time = np.array([history[e] for e in epochs])
    # print(probs_over_time)
    
    plt.figure(figsize=(12, 8))
    vocab_size = probs_over_time.shape[1]
    
    for idx in range(vocab_size):
        token_name = idx
        plt.plot(epochs, probs_over_time[:, idx], label=f'Token {idx}: {token_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title(f'Evolution of Token Probabilities {config.vocab_list}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ImageLogits.png')

def get_current_temperature(epoch, total_epochs, initial_temp=1.0, final_temp=0.1, plateau_fraction=0):
    plateau_epochs = int(total_epochs * plateau_fraction)    
    if epoch < plateau_epochs:
        return initial_temp
    else:
        decay_epoch = epoch - plateau_epochs
        decay_total = total_epochs - plateau_epochs
        return initial_temp - (initial_temp - final_temp) * (decay_epoch / decay_total)
    
def convert_label_to_logits(batch_labels):
    img_size = config.image_size
    kernel_size = model.image_transformer.pool_block.kernel_size
    stride = model.image_transformer.pool_block.stride
    output_size = ((img_size - kernel_size)//stride) + 1
    final_output_size = output_size**2 + 3

    labels = torch.zeros(len(batch_labels), final_output_size, config.output_dim)

    one_hots = F.one_hot(batch_labels + 1, num_classes=config.output_dim).float()
    labels[:, 1:] = one_hots.unsqueeze(1).expand(-1, final_output_size - 1, -1)
    return labels

token_history = {}

def train_model():
    model.train()  
    for epoch in range(config.num_epochs):
        current_temp = get_current_temperature(
            epoch, 
            config.num_epochs, 
            initial_temp=config.start_temp, 
            final_temp=config.end_temp
        )
        total_loss = 0.0
        for batch in config.train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)  # Move data to GPU

            # print(labels[0])

            labels = convert_label_to_logits(labels)

            # print(inputs[:5].shape)
            # print(labels.shape)
            # print(labels[0].argmax(-1))


            if config.logit_dataset:
                labels = labels.argmax(dim=-1) 

                outputs = model(inputs, temperature=current_temp) 

                mask = torch.zeros_like(outputs)
                mask[:, 1:, :] = 1
                outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()
                outputs = outputs_modified

                selected_outputs = outputs[:, 1:, :]  
                selected_labels = labels[:, 1:]       

                flattened_outputs = selected_outputs.contiguous().view(-1, outputs.size(-1)).to(config.device)
                flattened_labels = selected_labels.contiguous().view(-1).to(config.device)

                loss = criterion(flattened_outputs, flattened_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                log_token_preferences(model, epoch, token_history)

            else:
                ###WORKS GOOD
                # Remove the line: labels = labels.argmax(dim=-1)

                outputs = model(inputs, temperature=current_temp)

                mask = torch.zeros_like(outputs)
                mask[:, 1:, :] = 1
                outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()
                outputs = outputs_modified

                selected_outputs = outputs[:, 1:, :]  
                selected_labels = labels[:, 1:, :]    

                flattened_outputs = selected_outputs.contiguous().view(-1, selected_outputs.size(-1))
                flattened_labels = selected_labels.contiguous().view(-1, selected_labels.size(-1))

                log_probs = torch.nn.functional.log_softmax(flattened_outputs, dim=1)
                loss = torch.nn.functional.kl_div(log_probs, flattened_labels.to(config.device), reduction='batchmean')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                log_token_preferences(model, epoch, token_history)


                ###WORKS GOOD

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Temp: {current_temp:.3f}, Loss: {total_loss / len(config.train_loader):.4f}")
        evaluate_model()

def evaluate_model():
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in config.test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)  # Move data to GPU
            labels = convert_label_to_logits(labels)
            labels = labels.argmax(dim=-1).to(config.device)  

            outputs = model(inputs, temperature=config.temperature)  
            predicted = outputs.argmax(dim=-1)

            # print(predicted.shape)
            # print(labels.shape)

            correct += (predicted[:, 1:] == labels[:, 1:]).sum().item()
            total += labels[:, 1:].numel()

    print(f"Test Accuracy: {correct / total:.4f}")

train_model()
plot_token_preferences(token_history)

# #Visualize first batch
# output_first_batch = model(test_input.to(config.device))
# print(output_first_batch.shape)
# outputs_after_training = output_first_batch.argmax(dim=-1)

# cmap = ListedColormap(['black', 'red', 'green', 'blue'])
# num_samples_to_show = 25

# columns = 7 
# rows = math.ceil(num_samples_to_show / columns)
# fig, axs = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
# axs = axs.flatten()

# for i in range(num_samples_to_show):
#     original_img = test_input[i].squeeze().numpy()
#     axs[i].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
#     axs[i].set_title(f"GT: {test_labels[i]}, Before: {outputs_before_training[i][1]}, After: {outputs_after_training[i][1] - 1}", fontsize=8)
#     axs[i].axis('off')

# for ax in axs[num_samples_to_show:]:
#     ax.axis('off')

# fig.suptitle("Model Predictions", fontsize=16, y=0.99)
# plt.tight_layout()
# plt.show()

output_first_batch = model(test_input.to(config.device))
outputs_after_training = output_first_batch.argmax(dim=-1)

# cmap = ListedColormap(['black', 'red', 'green', 'blue'])
# num_samples_to_show = 25
# columns = 7
# rows = math.ceil(num_samples_to_show / columns)

# fig, axs = plt.subplots(rows * 2, columns, figsize=(2 * columns, 4 * rows))  # doubled the rows to show side-by-side vertically
# axs = axs.flatten()

# # Pass test images through transformer
# with torch.no_grad():
#     conv_out = model.image_transformer.conv_block(test_input[:num_samples_to_show].to(config.device))
#     pooled_out = model.image_transformer.pool_block(conv_out)

# for i in range(num_samples_to_show):
#     original_img = test_input[i].squeeze().cpu().numpy()
#     transformed_img = pooled_out[i].squeeze().cpu().numpy()

#     # Top row: Original
#     axs[i].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
#     axs[i].set_title(f"GT: {test_labels[i]}\nBefore: {outputs_before_training[i][1]}, After: {outputs_after_training[i][1] - 1}", fontsize=7)
#     axs[i].axis('off')

#     # Bottom row: Transformed
#     axs[i + num_samples_to_show].imshow(transformed_img, cmap='gray', interpolation='nearest')
#     axs[i + num_samples_to_show].set_title("Transformed", fontsize=7)
#     axs[i + num_samples_to_show].axis('off')

# # Hide any extra axes
# for ax in axs[2 * num_samples_to_show:]:
#     ax.axis('off')

# fig.suptitle("Original vs Transformed Images", fontsize=16, y=0.92)
# plt.tight_layout()
# plt.show()
cmap = ListedColormap(['black', 'red', 'green', 'blue'])
num_samples_to_show = 25
columns = 7
rows = math.ceil(num_samples_to_show / columns)

fig, axs = plt.subplots(rows, columns * 2, figsize=(2 * columns * 2, 4 * rows))  # doubled columns for side-by-side
axs = axs.reshape(rows, columns * 2)

# Get transformed output
with torch.no_grad():
    conv_out = model.image_transformer.conv_block(test_input[:num_samples_to_show].to(config.device))
    pooled_out = model.image_transformer.pool_block(conv_out)

for idx in range(num_samples_to_show):
    row = idx // columns
    col = (idx % columns) * 2  # Double column spacing for side-by-side

    # Original image
    original_img = test_input[idx].squeeze().cpu().numpy()
    axs[row, col].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    axs[row, col].set_title(
        f"GT: {test_labels[idx]}\nBefore: {outputs_before_training[idx][1]}, After: {outputs_after_training[idx][1] - 1}", 
        fontsize=7
    )
    axs[row, col].axis('off')

    # Transformed image
    transformed_img = pooled_out[idx].squeeze().cpu().numpy()
    axs[row, col + 1].imshow(transformed_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    axs[row, col + 1].set_title("Transformed", fontsize=7)
    axs[row, col + 1].axis('off')

# Hide any unused subplots
total_axes = rows * columns * 2
for ax in axs.flatten()[2 * num_samples_to_show:]:
    ax.axis('off')

fig.suptitle("Original and Transformed Images Side by Side", fontsize=16, y=0.95)
plt.tight_layout()
plt.show()

os.system(r'find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf')


# # 2. Test the data loaders
# train_loader = config.train_loader
# test_loader = config.test_loader

# print(f"Training batches: {len(train_loader)}")
# print(f"Test batches: {len(test_loader)}\n")

# # 3. Inspect first batch
# sample_batch = next(iter(train_loader))
# inputs, labels = sample_batch
# print(f"Batch shape - inputs: {inputs.shape}, labels: {labels.shape}")
# print(f"Sample input shape: {inputs[0].shape}")
# print(f"Sample label: {labels[0]}\n")

# # 4. Fetch one batch (again, just to illustrate)
# inputs, labels = next(iter(train_loader))

# num_samples_to_show = 5
# cmap = ListedColormap(['black', 'red', 'green', 'blue'])

# # 5. Reference to the original ImageDataset (from the Subset in train_loader)
# dataset_ref = train_loader.dataset.dataset  

# # 6. Process and store the first five images in the batch
# processed_images = []
# for i in range(num_samples_to_show):
#     np_img = inputs[i].squeeze().numpy()  # [32, 32]
#     processed = dataset_ref._process_image(np_img)
#     processed_images.append(processed)

# # 7. Display original vs processed images
# fig, axs = plt.subplots(2, num_samples_to_show, figsize=(3 * num_samples_to_show, 6))

# for i in range(num_samples_to_show):
#     # Top row: original image
#     original_img = inputs[i].squeeze().numpy()
#     axs[0, i].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
#     axs[0, i].set_title(f"Original: {labels[i].item()}")
#     axs[0, i].axis('off')

#     # Bottom row: processed image
#     proc_img = processed_images[i].detach().numpy()
#     axs[1, i].imshow(proc_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
#     axs[1, i].set_title(f"Processed: {labels[i].item()}")
#     axs[1, i].axis('off')

# plt.tight_layout()
# plt.savefig('dataset_test_1D_IMGAGE.png')


