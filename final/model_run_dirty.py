# import sys
# sys.path.append('/home/ruov/projects/AlgorithmicCounting/srcII/')

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import os
import math
from torch import nn, optim
from matplotlib.colors import ListedColormap
# from AlgorithmicCounting1DImage import config1DImage
# from AlgorithmicCounting1DImage.Model import model_1D
# from AlgorithmicCounting1DImage.Utils import RASPModel, dataset_creator, dataset_creator_image

class Config:
    def __init__(self):
        # Model structure
        self.n_layers = 1

        # Special tokens
        self.bos = "BOS"
        self.sep = "SEP"
        self.tmp = "TMP"

        # Length limits
        self.max_rasp_len = 40

        # MLP exactness (experimental)
        self.mlp_exactness = 1_000_000

        # Vocabulary
        self.vocab_req = {"SEP", "TMP"}
        self.vocab_tokens = {'0', '1', '2', '3'}
        self.vocab = self.vocab_tokens.union(self.vocab_req)

        # Counting settings
        self.token_to_count = '1'
        self.index_to_count = 1  # 1 for first token, -1 for last token

        # Test inputs
        self.test_input_listI = [
            ["BOS","1","SEP","1","1","2","2","1","2"],
            ["BOS","1","SEP","1","1","2","2","1","2"],
        ]
        self.test_input_listII = [
            ["BOS","TMP","SEP","1","1","2","2","1","2"],
            ["BOS","TMP","SEP","1","1","2","2","1","2"],
        ]
        self.test_input_list_ImageI = [
            ["BOS","TMP","SEP","1","1","2","1","1","1"],
            ["BOS","TMP","SEP","1","1","2","1","1","1"],
        ]
        self.test_input_list_ImageII = [
            ["BOS","TMP","SEP","0","1","1","0","1","1"],
            ["BOS","TMP","SEP","0","1","1","0","1","1"],
        ]
        self.test_input_list_ImageIII = [
            ["BOS","TMP","SEP","0","1","1","1","0","1"],
            ["BOS","TMP","SEP","0","1","1","1","0","1"],
        ]
        self.test_input_list_ImageIV = [
            ["BOS","TMP","SEP","0","1","1","1","0","0"],
            ["BOS","TMP","SEP","0","1","1","1","0","0"],
        ]
        self.test_input_list_ImageV = [
            ["BOS","TMP","SEP","0","0","0","0","0","0"],
            ["BOS","TMP","SEP","0","0","0","0","0","0"],
        ]
        self.test_input_list_ImageVI = [
            ["BOS","TMP","SEP","1","0","1","0","1","0"],
            ["BOS","TMP","SEP","1","0","1","0","1","0"],
        ]
        self.diff_size_input_list = [
            ["BOS","TMP","SEP","1","1","2","2","1","2","2","2","1","2"]
        ]
        self.diff_size_input_listII = [
            ["BOS","TMP","SEP","2","1","0"]
        ]

        # Dataset flags
        self.logit_dataset = False

        # Training hyperparameters
        self.batch_size = 1000
        self.labels_samples = 7
        self.n_samples = 1500
        self.train_split = 0.9

        # Data loaders & device
        self.train_loader = None
        self.test_loader = None
        self.device = None

        # Optimization
        self.learning_rate = 1e-2
        self.num_epochs = 200

        # Temperature schedule
        self.temperature = 0.1
        self.start_temp = 0.1
        self.end_temp = 0.001

        # RASP placeholders
        self.rasp_func = None
        self.rasp_model = None
        self.out_rasp = None

        # Model internals
        self.model_config = None
        self.model_parameters = None
        self.embedding_dim = None
        self.output_dim = None
        self.unembed_matrix = None
        self.encoding_func = None

        # Vocabulary transforms
        self.vocab_list = None
        self.encoded_vocab = None
        self.embedded_vocab = None

        # Batch size for experiments
        self.experiment_batch_size = 20

config1DImage = Config()

import torch
import torch.nn as nn
# from AlgorithmicCounting1DImage.ModelComponent import embedding
# from AlgorithmicCounting1DImage.ModelComponent import transformer
# from AlgorithmicCounting1DImage.ModelComponent import unembedding
# from AlgorithmicCounting1DImage.Classifier import image_transformer
# from AlgorithmicCounting1DImage.Classifier import embedder_classifier


class Model_1DImage(nn.Module):
    def __init__(self, model_config, model_parameters, unembedding_data, encoding_func, encoded_vocab):
        super(Model_1DImage, self).__init__()

        self.image_transformer = Image_Transformer()

        self.embedding_dim = unembedding_data.shape[0]

        self.encoder = encoding_func

        # self.embedder = embedding.Embedding_Module(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
        #                                                           token_embed_data=model_parameters['embeddings']['token_embed']['embeddings'],
        #                                                           trainable=False).to(config1DImage.device)

        self.classifier = Classifier(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings']).to(config1DImage.device)
        
        self.transformer = Transformer_Module(model_config=model_config,
                                                                         model_parameters=model_parameters,
                                                                         trainable=False,
                                                                         )
        
        self.unembedder = Unembedding_Module(unembedding_data=unembedding_data,
                                                                        use_unembed_argmax=False,
                                                                        trainable=False)
        # self.embedder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.unembedder.requires_grad_(False)
            
    def forward(self, x, temperature=0.1):
        x = self.image_transformer(x)
        x = self.encoder(x)        
        x = self.classifier(x, temperature=temperature)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
    def foward_without_classification(self, x):
        x = self.image_transformer(x)
        x = self.encoder(x)    
        x = self.embedder(x)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
import jax
import torch
from tracr.rasp import rasp
from tracr.compiler import compiling
# from AlgorithmicCounting1DImage import config1DImage
# from AlgorithmicCounting1DImage.Utils import utils

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from AlgorithmicCounting1DImage import config1DImage

class Classifier(nn.Module):
    def __init__(self, pos_embed_data, token_embed_data):
        super(Classifier, self).__init__()

        self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(pos_embed_data.copy()), dtype=torch.float32),freeze=True)
        self.token_embed = nn.Embedding.from_pretrained(torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32),freeze=True)
        self.token_embed_matrix = torch.tensor(np.array(token_embed_data.copy()), dtype=torch.float32)
        self.vocab_size = self.token_embed_matrix.shape[0]   
        self.token_logits = nn.Parameter(torch.zeros(self.vocab_size))

        # self.edge_cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, padding=1)

        # if config1DImage.fixed_CNN:
        #     with torch.no_grad():
        #         self.edge_cnn.weight.copy_(torch.tensor([[[-1.0, 1.0]]], dtype=torch.float32))
        #         if self.edge_cnn.bias is not None:
        #             self.edge_cnn.bias.zero_()
        #     self.edge_cnn.requires_grad_(False)


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
        # seperator_token = token_embeddings[:, 2, :].unsqueeze(1)
        # learned_token_to_count = self.token_embed_matrix[1].expand(batch_size, -1).to(x.device).unsqueeze(1)

        # cos_sim = F.cosine_similarity(token_embeddings, learned_token_to_count, dim=-1)
        # threshold = 0.9
        # mask = (cos_sim >= threshold).float().unsqueeze(-1)

        # print(f'Mask before CNN: {mask[0].T}')
        # mask_temp = mask.transpose(1, 2)  
        # mask_temp = self.edge_cnn(mask_temp)
        # mask_edge = mask_temp.transpose(1, 2)[:,:-1]
        # mask_edge = mask_edge.clamp(min=0)
        # diff_mask = mask - mask_edge
        # print(f'Mask after CNN:  {mask_edge[0].T}')
        # print(f'Diff after CNN:  {diff_mask[0].T}')

        # modified_embeddings = token_embeddings * (1 - diff_mask) + seperator_token * diff_mask 
        token_embeddings[:, 1, :] = learned_token_to_count.squeeze(1)       
        position_embeddings = self.pos_embed(torch.arange(x.shape[-1], device=x.device))
        return (token_embeddings + position_embeddings).float()



import torch
import torch.nn as nn
# from AlgorithmicCounting1DImage.ModelComponent import mlp
# from AlgorithmicCounting1DImage.ModelComponent import attention

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
            self.layers = []
            for _ in range(n_layers):
                ATT_Layer = attention.ATT_Module(key_size = model_config.key_size, 
                                                num_heads=model_config.num_heads,
                                                query_data=model_parameters['layers'][0]['attn']['query'],
                                                key_data=model_parameters['layers'][0]['attn']['key'],
                                                value_data=model_parameters['layers'][0]['attn']['value'],
                                                linear_data=model_parameters['layers'][0]['attn']['linear'],
                                                trainable=True)
                self.layers.append(ATT_Layer)

                MLP_Layer = mlp.MLP_Module(activation_function=model_config.activation_function, 
                                        linear_1_data=model_parameters['layers'][0]['mlp']['linear_1'],
                                        linear_2_data=model_parameters['layers'][0]['mlp']['linear_2'],
                                        trainable=True)
                self.layers.append(MLP_Layer)

                self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x
    
import torch                                                                                                                              
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from AlgorithmicCounting1DImage import config1DImage
from matplotlib.colors import ListedColormap


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

        x_reshaped = x.view(x.shape[0], -1)
        x_list = x_reshaped.tolist()

        prefix = [config1DImage.bos, config1DImage.tmp, config1DImage.sep]

        x_list_str = []
        for row in x_list:
            row = [int(token) for token in row]
            row_str = list(map(str, row))
            new_row = prefix + row_str
            x_list_str.append(new_row)

        return x_list_str


    
import torch
import numpy as np
import torch.nn as nn

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


import torch
import numpy as np
import torch.nn as nn

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
    
import torch
import numpy as np
import torch.nn as nn

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
    
jax.config.update("jax_default_matmul_precision", "highest")

class Models:
    def __init__(self):
        pass

    def count_agnostic_first(self):
        SELECT_ALL_TRUE = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        LENGTH = rasp.SelectorWidth(SELECT_ALL_TRUE) * 0
        SELECT_FIRST = rasp.Select(rasp.indices, LENGTH , rasp.Comparison.EQ)
        FIRST_TOKEN = rasp.Aggregate(SELECT_FIRST, rasp.tokens)
        COUNT = rasp.SelectorWidth(rasp.Select(rasp.tokens, FIRST_TOKEN, rasp.Comparison.EQ))
        return COUNT

    def count_agnostic_last(self):
        SELECT_ALL_TRUE = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        LENGTH = rasp.SelectorWidth(SELECT_ALL_TRUE) 
        SELECT_FIRST = rasp.Select(rasp.indices, LENGTH - 1, rasp.Comparison.EQ)
        FIRST_TOKEN = rasp.Aggregate(SELECT_FIRST, rasp.tokens)
        COUNT = rasp.SelectorWidth(rasp.Select(rasp.tokens, FIRST_TOKEN, rasp.Comparison.EQ))
        return COUNT
    
    def compile_model(self):
        config1DImage.rasp_model = compiling.compile_rasp_to_model(
            config1DImage.rasp_func,
            vocab=config1DImage.vocab,
            max_seq_len=config1DImage.max_rasp_len,
            compiler_bos=config1DImage.bos,
            # compiler_pad=config.pad,
            mlp_exactness=config1DImage.mlp_exactness
        )    

def encoding_func(x):
    encoded_samples = []
    for sample in x:
        if len(sample) < 3:
            raise Exception("Something went wrong with the dataset") 
        else:
            sample = torch.tensor(config1DImage.rasp_model.custom_encode(sample), dtype=torch.int64) 
            encoded_samples.append(sample)

    return torch.stack(encoded_samples).to(config1DImage.device)

def CompileRaspModel():
    model_class = Models()
    config1DImage.rasp_func = model_class.count_agnostic_first()
    model_class.compile_model()
    config1DImage.out_rasp = config1DImage.rasp_model.apply(config1DImage.test_input_listI[0])
    print(f"Count RASP token: '{str(config1DImage.test_input_listI[0][config1DImage.index_to_count])}' expected: {str(config1DImage.test_input_listI[0].count(config1DImage.test_input_listI[0][config1DImage.index_to_count]))}, computed: {str(config1DImage.out_rasp.decoded[-1])}, raw out: {config1DImage.out_rasp.decoded}")

    config1DImage.model_config = config1DImage.rasp_model.model_config
    config1DImage.model_config.activation_function = torch.nn.ReLU()
    config1DImage.model_parameters = extract_weights(config1DImage.rasp_model.params)

    config1DImage.unembed_matrix = config1DImage.out_rasp.unembed_matrix
    config1DImage.encoding_func = encoding_func
    config1DImage.embedding_dim = config1DImage.unembed_matrix.shape[0]
    config1DImage.output_dim = config1DImage.unembed_matrix.shape[1]

    config1DImage.vocab_list = [[config1DImage.bos] + list(config1DImage.vocab)]
    config1DImage.encoded_vocab = config1DImage.encoding_func(config1DImage.vocab_list)

    return 

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

import sys
sys.path.append('/home/ruov/projects/AlgorithmicCounting/srcII/')

import os
import torch
import numpy as np
import torch.nn as nn
from skimage.draw import polygon
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
# from AlgorithmicCounting1DImage import config1DImage

matplotlib.use('TkAgg')  # or another interactive backend, such as Qt5Agg

# # Configuration class (mock - replace with your actual config)
# class Config:
#     def __init__(self):
#         self.batch_size = 32
#         self.train_split = 0.8
#         self.output_dim = 7  # For 0-6 red objects classification
#         self.max_seq_len = 256  # Not used but kept for compatibility

# config1DImage = Config()

# Image generation parameters
IMAGE_SIZE = 32
SAFE_MARGIN = 3

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
    """
    Generates an image containing:
      - n_red_objects red shapes (squares, triangles, or diamonds)
      - num_distractors distractors (squares, triangles with random orientations, or diamonds)
      - num_extra_triangles additional random-position triangles (green/blue)
      - num_diamonds additional random-position diamonds (green/blue)

    Returns:
      A 2D NumPy array (32 x 32) with color-coded integers:
        0 = background (black)
        1 = red
        2 or 3 = distractor color (green/blue)
    """
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # --- PLACE N RED OBJECTS ---
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

    # --- PLACE DISTRACTOR SHAPES (could be squares, triangles with orientations, diamonds) ---
    for _ in range(num_distractors):
        shape_type = np.random.choice(["square", "triangle", "diamond"])
        color = np.random.choice([2, 3])  # 2 = green, 3 = blue, for example

        if shape_type == "square":
            size = np.random.randint(2, 7)
            top = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - size - SAFE_MARGIN)
            left = np.random.randint(SAFE_MARGIN, IMAGE_SIZE - size - SAFE_MARGIN)
            if can_place_shape(image, top, left, size, size):
                image[top:top + size, left:left + size] = color

        elif shape_type == "triangle":
            # Triangles with random orientation
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

            # Check placement
            if can_place_polygon(image, r_coords, c_coords):
                rr, cc = polygon(r_coords, c_coords)
                # Ensure no overlap
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

    # --- PLACE EXTRA TRIANGLES ---
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

    # --- PLACE DIAMONDS ---
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

class ImageDataset(Dataset):
    def __init__(self, num_samples_per_label=10, labels=range(7)):
        """
        Creates a dataset where label = number of red objects.
        For each label, num_samples_per_label images are generated.
        """
        super().__init__()
        self.data = []
        self.labels = []
        
        # Define processing pipeline
        self.kernels = [
            torch.tensor([[[-1., 1.]]]).unsqueeze(0),
            torch.tensor([[[-1.], [1.]]]).unsqueeze(0),
            torch.tensor([[[0., 1.], [-1., 0.]]]).unsqueeze(0),
            torch.tensor([[[1., 0.], [0., -1.]]]).unsqueeze(0),
        ]
        self.pool = nn.MaxPool2d(kernel_size=6, stride=6)
        
        # Generate dataset
        for label in labels:
            for _ in range(num_samples_per_label):
                img, actual_label = generate_image(label)
                processed = torch.from_numpy(img)
                self.data.append(processed)
                self.labels.append(actual_label)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    return torch.stack(inputs).unsqueeze(1), torch.stack(labels)

def GenerateDataset():
    num_samples_per_label = config1DImage.n_samples // config1DImage.labels_samples
    full_dataset = ImageDataset(num_samples_per_label=num_samples_per_label, labels=range(config1DImage.labels_samples))
    train_size = int(config1DImage.train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    config1DImage.train_loader = DataLoader(
        train_dataset,
        batch_size=config1DImage.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    config1DImage.test_loader = DataLoader(
        test_dataset,
        batch_size=config1DImage.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

# Check for GPU availability
config1DImage.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {config1DImage.device}")

# Compile RASP and compute dataset
CompileRaspModel()
GenerateDataset()

# Move model to GPU
model = Model_1DImage(model_config=config1DImage.model_config,
                          model_parameters=config1DImage.model_parameters,
                          unembedding_data=config1DImage.unembed_matrix,
                          encoding_func=config1DImage.encoding_func,
                          encoded_vocab=config1DImage.encoded_vocab
                          ).to(config1DImage.device)

#Visualize first batch
test_input, test_labels = next(iter(config1DImage.test_loader))
output_first_batch = model(test_input.to(config1DImage.device))
print(output_first_batch.shape)
outputs_before_training = output_first_batch.argmax(dim=-1)

# cmap = ListedColormap(['black', 'red', 'green', 'blue'])
# num_samples_to_show = 25

# columns = 7 
# rows = math.ceil(num_samples_to_show / columns)
# fig, axs = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
# axs = axs.flatten()

# for i in range(num_samples_to_show):
#     original_img = test_input[i].squeeze().numpy()
#     axs[i].imshow(original_img.astype(int), cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
#     axs[i].set_title(f"GT: {test_labels[i]}, PRED: {outputs[i][1]}", fontsize=8)
#     axs[i].axis('off')

# for ax in axs[num_samples_to_show:]:
#     ax.axis('off')

# fig.suptitle("Model Predictions Untrained", fontsize=16, y=0.99)
# plt.tight_layout()
# plt.show()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config1DImage.learning_rate)

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
    plt.title(f'Evolution of Token Probabilities {config1DImage.vocab_list}')
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
    img_size = IMAGE_SIZE
    kernel_size = model.image_transformer.pool_block.kernel_size
    stride = model.image_transformer.pool_block.stride
    output_size = ((img_size - kernel_size)//stride) + 1
    final_output_size = output_size**2 + 3

    labels = torch.zeros(len(batch_labels), final_output_size, config1DImage.output_dim)

    one_hots = F.one_hot(batch_labels + 1, num_classes=config1DImage.output_dim).float()
    labels[:, 1:] = one_hots.unsqueeze(1).expand(-1, final_output_size - 1, -1)
    return labels

token_history = {}

def train_model():
    model.train()  
    for epoch in range(config1DImage.num_epochs):
        current_temp = get_current_temperature(
            epoch, 
            config1DImage.num_epochs, 
            initial_temp=config1DImage.start_temp, 
            final_temp=config1DImage.end_temp
        )
        total_loss = 0.0
        for batch in config1DImage.train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config1DImage.device), labels.to(config1DImage.device)  # Move data to GPU

            # print(labels[0])

            labels = convert_label_to_logits(labels)

            # print(inputs[:5].shape)
            # print(labels.shape)
            # print(labels[0].argmax(-1))


            if config1DImage.logit_dataset:
                labels = labels.argmax(dim=-1) 

                outputs = model(inputs, temperature=current_temp) 

                mask = torch.zeros_like(outputs)
                mask[:, 1:, :] = 1
                outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()
                outputs = outputs_modified

                selected_outputs = outputs[:, 1:, :]  
                selected_labels = labels[:, 1:]       

                flattened_outputs = selected_outputs.contiguous().view(-1, outputs.size(-1)).to(config1DImage.device)
                flattened_labels = selected_labels.contiguous().view(-1).to(config1DImage.device)

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
                loss = torch.nn.functional.kl_div(log_probs, flattened_labels.to(config1DImage.device), reduction='batchmean')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                log_token_preferences(model, epoch, token_history)

                ###WORKS GOOD

        print(f"Epoch {epoch + 1}/{config1DImage.num_epochs}, Temp: {current_temp:.3f}, Loss: {total_loss / len(config1DImage.train_loader):.4f}")
        evaluate_model()

def evaluate_model():
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in config1DImage.test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config1DImage.device), labels.to(config1DImage.device)  # Move data to GPU
            labels = convert_label_to_logits(labels)
            labels = labels.argmax(dim=-1).to(config1DImage.device)  

            outputs = model(inputs, temperature=config1DImage.temperature)  
            predicted = outputs.argmax(dim=-1)

            # print(predicted.shape)
            # print(labels.shape)

            correct += (predicted[:, 1:] == labels[:, 1:]).sum().item()
            total += labels[:, 1:].numel()

    print(f"Test Accuracy: {correct / total:.4f}")

train_model()
plot_token_preferences(token_history)

# #Visualize first batch
# output_first_batch = model(test_input.to(config1DImage.device))
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

output_first_batch = model(test_input.to(config1DImage.device))
outputs_after_training = output_first_batch.argmax(dim=-1)

# cmap = ListedColormap(['black', 'red', 'green', 'blue'])
# num_samples_to_show = 25
# columns = 7
# rows = math.ceil(num_samples_to_show / columns)

# fig, axs = plt.subplots(rows * 2, columns, figsize=(2 * columns, 4 * rows))  # doubled the rows to show side-by-side vertically
# axs = axs.flatten()

# # Pass test images through transformer
# with torch.no_grad():
#     conv_out = model.image_transformer.conv_block(test_input[:num_samples_to_show].to(config1DImage.device))
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
    conv_out = model.image_transformer.conv_block(test_input[:num_samples_to_show].to(config1DImage.device))
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
# train_loader = config1DImage.train_loader
# test_loader = config1DImage.test_loader

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

