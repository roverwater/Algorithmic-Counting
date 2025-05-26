import sys
# sys.path.append('/home/ruov/projects/AlgorithmicCounting/srcII/')

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn, optim
# from AlgorithmicCounting1D import config
# from AlgorithmicCounting1D.Model import model_1D
# from AlgorithmicCounting1D.Utils import RASPModel, dataset_creator

from torchviz import make_dot
from torchsummary import summary

class Config:
    def __init__(self):
        # Model architecture
        self.n_layers = 1

        # Special tokens
        self.bos = "BOS"
        self.sep = "SEP"
        self.tmp = "TMP"
        self.custom_pad = self.tmp

        # Generation settings
        self.max_rasp_len = 38
        self.mlp_exactness = 1_000_000

        # Vocabulary
        self.vocab_req = {self.sep, self.tmp}
        self.vocab_tokens = {'0', '1', '2', '3'}
        self.vocab = self.vocab_tokens.union(self.vocab_req)

        # Counting settings
        self.token_to_count = '1'
        self.index_to_count = 1  # 1 for first token or -1 for last token

        # Test inputs
        self.test_input_listI = [
            [self.bos, '1', self.sep, '1', '1', '2', '2', '1', '2']
        ]
        self.test_input_listII = [
            [self.bos, self.tmp, self.sep, '1', '1', '2', '2', '1', '2']
        ]
        self.diff_size_input_list = [
            [self.bos, self.tmp, self.sep, '1', '1', '2', '2', '1', '2', '2', '2', '1', '2']
        ]
        self.diff_size_input_listII = [
            [self.bos, self.tmp, self.sep, '2', '1', '0']
        ]

        # Dataset flags
        self.logit_dataset = False

        # Data loader settings
        self.batch_size = 1000
        self.max_seq_len = 28
        self.train_split = 0.9
        self.train_loader = None
        self.test_loader = None
        self.device = None

        # Optimization
        self.learning_rate = 1e-1
        self.num_epochs = 1000
        self.temperature = 0.1
        self.start_temp = 0.3
        self.end_temp = 0.01

        # RASP components (to be set later)
        self.rasp_func = None
        self.rasp_model = None
        self.out_rasp = None

        # Model internals (to be set later)
        self.model_config = None
        self.model_parameters = None
        self.embedding_dim = None
        self.output_dim = None
        self.unembed_matrix = None
        self.encoding_func = None

        # Vocabulary processing (to be set later)
        self.vocab_list = None
        self.encoded_vocab = None
        self.embedded_vocab = None

        # Experiment settings
        self.experiment_batch_size = 20
        self.counts_dis = None


        self.used_seed = None

config = Config()

import jax
import torch
from tracr.rasp import rasp
from tracr.compiler import compiling
# from AlgorithmicCounting1D import config
# from AlgorithmicCounting1D.Utils import utils

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
        config.rasp_model = compiling.compile_rasp_to_model(
            config.rasp_func,
            vocab=config.vocab,
            max_seq_len=config.max_rasp_len,
            compiler_bos=config.bos,
            # compiler_pad=config.pad,
            mlp_exactness=config.mlp_exactness
        )    

def encoding_func(x):
    encoded_samples = []
    for sample in x:
        if len(sample) < 3:
            raise Exception("Something went wrong with the dataset") 
        else:
            sample = torch.tensor(config.rasp_model.custom_encode(sample), dtype=torch.int64) 
            encoded_samples.append(sample)

    return torch.stack(encoded_samples).to(config.device)

def CompileRaspModel():
    model_class = Models()
    config.rasp_func = model_class.count_agnostic_first()
    model_class.compile_model()
    config.out_rasp = config.rasp_model.apply(config.test_input_listI[0])
    print(f"Count RASP token: '{str(config.test_input_listI[0][config.index_to_count])}' expected: {str(config.test_input_listI[0].count(config.test_input_listI[0][config.index_to_count]))}, computed: {str(config.out_rasp.decoded[-1])}, raw out: {config.out_rasp.decoded}")

    config.model_config = config.rasp_model.model_config
    config.model_config.activation_function = torch.nn.ReLU()
    config.model_parameters = extract_weights(config.rasp_model.params)

    config.unembed_matrix = config.out_rasp.unembed_matrix
    config.encoding_func = encoding_func
    config.embedding_dim = config.unembed_matrix.shape[0]
    config.output_dim = config.unembed_matrix.shape[1]

    config.vocab_list = [[config.bos] + list(config.vocab)]
    config.encoded_vocab = config.encoding_func(config.vocab_list)

    return 

class Model_1D(nn.Module):
    def __init__(self, model_config, model_parameters, unembedding_data, encoding_func, encoded_vocab):
        super(Model_1D, self).__init__()

        self.embedding_dim = unembedding_data.shape[0]

        self.encoder = encoding_func

        self.embedder = Embedding_Module(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings'],
                                                                  trainable=False).to(config.device)

        self.classifier = StaticTokenReplacer(token_embeddings=self.embedder.token_embed.weight.data, pos_embed_layer=self.embedder.pos_embed)
        
        self.transformer = Transformer_Module(model_config=model_config,
                                                                         model_parameters=model_parameters,
                                                                         trainable=False,
                                                                         )
        
        self.unembedder = Unembedding_Module(unembedding_data=unembedding_data,
                                                                        use_unembed_argmax=False,
                                                                        trainable=False)
        self.embedder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.unembedder.requires_grad_(False)
            
    def forward(self, x, temperature=0.1):
        x = self.encoder(x)        
        x = self.embedder(x)
        x = self.classifier(x, temperature=temperature)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
    def foward_without_classification(self, x):
        x = self.encoder(x)    
        x = self.embedder(x)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x
    
class StaticTokenReplacer(nn.Module):
    def __init__(self, token_embeddings, pos_embed_layer):
        super().__init__()
        # Token embeddings (excluding BOS)
        # self.token_embeddings = token_embeddings[1:, :]  # [vocab_size-1, d_model]
        self.token_embeddings = token_embeddings

        self.vocab_size = self.token_embeddings.shape[0]
        
        # Positional embedding layer (critical!)
        self.pos_embed = pos_embed_layer
        
        # Learnable logits
        self.token_logits = nn.Parameter(torch.zeros(self.vocab_size))
        
    def forward(self, x, temperature, tmp_position_idx=1):
        batch_size = x.shape[0]
        
        # 1. Predict replacement token embedding (without positional)
        soft_weights = F.gumbel_softmax(
            self.token_logits.unsqueeze(0).expand(batch_size, -1),
            tau=temperature,
            hard=False
        )
        replacement_token_emb = torch.einsum("bv,vd->bd", soft_weights, self.token_embeddings)
        
        # 2. Add positional encoding for TMP's position
        pos_emb = self.pos_embed(torch.tensor(tmp_position_idx, device=x.device))  # [d_model]
        replacement_emb = replacement_token_emb + pos_emb.unsqueeze(0)  # [batch, d_model]
        
        # 3. Replace TMP token
        x_modified = x.clone()
        x_modified[:, tmp_position_idx, :] = replacement_emb
        
        return x_modified

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
import torch.nn as nn
# from AlgorithmicCounting1D.ModelComponent import mlp
# from AlgorithmicCounting1D.ModelComponent import attention

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

import torch
import numpy as np
# from AlgorithmicCounting1D import config
from torch.utils.data import Dataset, DataLoader, random_split

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class SequenceDataset(Dataset):
    def __init__(self, load_from_file=False, filename='dataset.pth'):
        if load_from_file:
            # Load data from file
            loaded_data = torch.load(filename)
            
            # Verify config matches saved parameters
            if (loaded_data['batch_size'] != config.batch_size or
                loaded_data['max_seq_len'] != config.max_seq_len or
                loaded_data['output_dim'] != config.output_dim):
                raise ValueError("Config parameters don't match saved dataset. Delete file to regenerate.")
            
            self.batch_size = loaded_data['batch_size']
            self.max_seq_len = loaded_data['max_seq_len']
            self.output_dim = loaded_data['output_dim']
            self.data = loaded_data['data']
            self.labels = loaded_data['labels']
            self.counts_dis = loaded_data['counts_dis']
            
            # Update config with loaded counts
            config.counts_dis = self.counts_dis
        else:
            # Original initialization logic
            self.batch_size = config.batch_size
            self.max_seq_len = config.max_seq_len
            self.output_dim = config.output_dim
            self.data = []
            self.labels = []
            self.counts_dis = []

            alpha = 1.8
            K = self.max_seq_len - 3
            ks = np.arange(0, K+1)
            weights = (ks + 1) ** (-alpha)
            weights /= weights.sum()

            for _ in range(self.batch_size):
                seq_len = self.max_seq_len
                num_special = np.random.choice(ks, p=weights)
                
                specials = [config.token_to_count] * int(num_special)
                others = [
                    str(np.random.choice(list(config.vocab_tokens), replace=True))
                    for _ in range(K - int(num_special))
                ]
                
                rest = specials + others
                np.random.shuffle(rest)
                sequence = [config.bos, config.tmp, config.sep] + rest
                
                while len(sequence) < self.max_seq_len:
                    sequence.append(config.custom_pad)
                
                self.data.append(sequence)
                count_a = sequence.count(config.token_to_count) + 1

                ground_truth = torch.zeros(self.max_seq_len, self.output_dim)
                if count_a < self.output_dim:
                    ground_truth[1:, count_a] = 1
                ground_truth[0, 28] = 1
                
                self.labels.append(ground_truth)
                self.counts_dis.append(count_a - 1)

            config.counts_dis = self.counts_dis

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def save_to_file(self, filename='dataset.pth'):
        data_to_save = {
            'batch_size': self.batch_size,
            'max_seq_len': self.max_seq_len,
            'output_dim': self.output_dim,
            'data': self.data,
            'labels': self.labels,
            'counts_dis': self.counts_dis
        }
        torch.save(data_to_save, filename)

def collate_fn(batch):
    inputs, labels = zip(*batch)  
    return list(inputs), torch.stack(labels)
    

def GenerateDataset():
    filename = 'dataset.pth'
    
    if os.path.exists(filename):
        print("Loading dataset from file...")
        dataset = SequenceDataset(load_from_file=True, filename=filename)
    else:
        print("Generating new dataset...")
        dataset = SequenceDataset()
        dataset.save_to_file(filename)
    
    # Use fixed seed for consistent splits
    # print(torch.seed)
    config.used_seed = torch.seed()
    # print(config.used_seed)
    # torch.manual_seed(42)
    train_size = int(config.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    config.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                   shuffle=True, collate_fn=collate_fn)
    config.test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                                  shuffle=False, collate_fn=collate_fn)

    # Print first batch from each loader
    print("\nTrain Data:")
    for batch in config.train_loader:
        inputs, labels = batch
        print(f"First batch contains {len(inputs)} sequences")
        break

    print("\nTest Data:")
    for batch in config.test_loader:
        inputs, labels = batch
        print(f"First batch contains {len(inputs)} sequences")
        break

# Example usage:
# GenerateDataset()  # First run generates and saves
# GenerateDataset()  # Subsequent runs load from file




# Check for GPU availability
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {config.device}")

CompileRaspModel()

# Move model to GPU
model = Model_1D(model_config=config.model_config,
                          model_parameters=config.model_parameters,
                          unembedding_data=config.unembed_matrix,
                          encoding_func=config.encoding_func,
                          encoded_vocab=config.encoded_vocab
                          ).to(config.device)

# print("\nTrainable Parameters in the Model:")
# for name, param in model.named_parameters():
#     print(f"{name}: Requires Grad = {param.requires_grad}")

print(model.embedder.token_embed.weight.data.shape)
print(model.parameters)
 
# raise Exception

# Print initial model outputs
print(f'With classifier {model(config.test_input_listI, temperature=config.temperature).argmax(-1)}')
print(f'Without classifier {model.foward_without_classification(config.test_input_listI).argmax(-1)}')
print(f'With classifier {model(config.test_input_listII, temperature=config.temperature).argmax(-1)}')
print(f'Without classifier {model.foward_without_classification(config.test_input_listII).argmax(-1)}')

GenerateDataset()

# Collect all input sequences from both train and test loaders
all_sequences = []

# Process training data
for batch in config.train_loader:
    inputs, _ = batch
    all_sequences.extend(inputs)

# Process test data
for batch in config.test_loader:
    inputs, _ = batch
    all_sequences.extend(inputs)

# Convert lists to tuples for hashing and count occurrences
from collections import Counter
sequence_counts = Counter(tuple(seq) for seq in all_sequences)

# Calculate duplicate statistics
duplicate_sequences = {seq: count for seq, count in sequence_counts.items() if count > 1}
num_duplicates = sum(count - 1 for count in duplicate_sequences.values())

print(f"Total samples: {len(all_sequences)}")
print(f"Unique sequences: {len(sequence_counts)}")
print(f"Duplicate sequences: {len(duplicate_sequences)}")
print(f"Total duplicate instances: {num_duplicates}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

def log_token_preferences(model, epoch, history):
    """
    Logs the softmax probabilities of the token_logits at the current epoch.
    
    Args:
        model: Instance of StaticTokenReplacer.
        epoch: Current epoch (or iteration) number.
        history: A dict to store probabilities across epochs.
    """
    # Compute token probabilities
    token_probs = F.softmax(model.classifier.token_logits, dim=0).detach().cpu().numpy()
    history[epoch] = token_probs

def plot_token_preferences(history):
    """
    Plots the evolution of all token probabilities over time.
    
    Args:
        history: A dict mapping epoch numbers to token probability arrays.
    """
    # Sorted epochs for proper plotting over time
    epochs = sorted(history.keys())
    
    # For debugging or verification, print the vocab mappings
    # print(f'V lst:      {config.vocab_list}')
    # print(f'Enc. V lst: {config.encoded_vocab}')
    # print(f'Emb. V lst: {config.embedded_vocab}')

    # Convert history into a numpy array with shape [num_epochs, vocab_size]
    probs_over_time = np.array([history[e] for e in epochs])
    print(probs_over_time)
    
    plt.figure(figsize=(12, 8))
    vocab_size = probs_over_time.shape[1]
    
    # Plot each token's probability over time with its name in the legend
    for idx in range(vocab_size):
        # token_idx = idx + 1 #shifted because of bos
        token_name = idx
        plt.plot(epochs, probs_over_time[:, idx], label=f'Token {idx}: {token_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title(f'Evolution of Token Probabilities {config.vocab_list}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('my_plot.png')

def get_current_temperature(epoch, total_epochs, initial_temp=1.0, final_temp=0.1):
    # Linear annealing
    # return config.temperature

    return initial_temp - (initial_temp - final_temp) * (epoch / total_epochs)

token_history = {}
loss_rasp_model = []
loss_cnn_model = []

def train_model():
    model.train()  
    for epoch in range(config.num_epochs):
        current_temp = get_current_temperature(
            epoch, 
            config.num_epochs, 
            initial_temp=0.1, 
            final_temp=0.1
        )
        total_loss = 0.0
        for batch in config.train_loader:
            inputs, labels = batch
            inputs, labels = inputs, labels.to(config.device)  # Move data to GPU

            if config.logit_dataset:
                labels = labels.argmax(dim=-1) 
                outputs = model(inputs, temperature=current_temp) 

                # MODIFIED: Focus only on position 1
                mask = torch.zeros_like(outputs)
                mask[:, 1, :] = 1  # Only TMP position contributes
                outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()
                
                # Directly select position 1
                selected_outputs = outputs_modified[:, 1, :]  # [batch, num_classes]
                selected_labels = labels[:, 1]                # [batch]

                loss = criterion(selected_outputs, selected_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                total_loss += loss.item()

                log_token_preferences(model, epoch, token_history)

            else:

                ###WORKS GOOD

                # Remove the line: labels = labels.argmax(dim=-1)

                outputs = model(inputs, temperature=current_temp)

                # Altered with correct loss
                mask = torch.zeros_like(outputs)
                mask[:, 1:, :] = 1
                outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()
                outputs = outputs_modified

                # Extract predictions and labels for tokens from the second token onward
                selected_outputs = outputs[:, 1:, :]  # Shape: [batch_size, seq_len-1, num_classes]
                selected_labels = labels[:, 1:, :]    # Shape: [batch_size, seq_len-1, num_classes]

                # Flatten the outputs and labels
                flattened_outputs = selected_outputs.contiguous().view(-1, selected_outputs.size(-1))
                flattened_labels = selected_labels.contiguous().view(-1, selected_labels.size(-1))

                # Compute log probabilities and KL divergence loss
                log_probs = torch.nn.functional.log_softmax(flattened_outputs, dim=1)
                loss = torch.nn.functional.kl_div(log_probs, flattened_labels, reduction='batchmean')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                log_token_preferences(model, epoch, token_history)


                ###WORKS GOOD


            # make_dot(outputs, params=dict(model.named_parameters())).render("model_architecture", format="png")
            # break




            # #Original
            # loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            # #Altered
            # mask = torch.zeros_like(outputs)
            # mask[:, 1, :] = 1  # Focus on the second token across all batches

            # # Retain gradients only for the second token
            # outputs_modified = outputs * mask + (outputs * (1 - mask)).detach()

            # outputs = outputs_modified

            # # Compute loss (unchanged)
            # loss = criterion(
            #     outputs.view(-1, outputs.size(-1)),  # Shape: [1600*15, 39]
            #     labels.view(-1)                               # Shape: [1600*15]
            # )

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Temp: {current_temp:.3f}, Loss: {total_loss / len(config.train_loader):.4f}")
        loss_rasp_model.append(total_loss / len(config.train_loader))
        evaluate_model()

def evaluate_model():
    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in config.test_loader:
            inputs, labels = batch
            inputs, labels = inputs, labels.to(config.device)  # Move data to GPU
            labels = labels.argmax(dim=-1)  

            outputs = model(inputs, temperature=config.temperature)  
            predicted = outputs.argmax(dim=-1)

            correct += (predicted[:, 1:] == labels[:, 1:]).sum().item()
            total += labels[:, 1:].numel()

    print(f"Test Accuracy: {correct / total:.4f}")

train_model()
# plot_token_preferences(token_history)
# plt.show()

# plt.hist(config.counts_dis, bins='auto')
# plt.show()

print(f'After training')
print(f'With classifier {model(config.test_input_listI, temperature=config.temperature).argmax(-1)}')
print(f'Without classifier {model.foward_without_classification(config.test_input_listI).argmax(-1)}')
print(f'With classifier {model(config.test_input_listII, temperature=config.temperature).argmax(-1)}')
print(f'Without classifier {model.foward_without_classification(config.test_input_listII).argmax(-1)}')

print(f'Different Size')
print(f'With classifier {model(config.diff_size_input_list, temperature=config.temperature).argmax(-1)}')
config.diff_size_input_list[0][1] = config.token_to_count
print(f'Without classifier {model.foward_without_classification(config.diff_size_input_list).argmax(-1)}')
print(f'With classifier {model(config.diff_size_input_listII, temperature=config.temperature).argmax(-1)}')
config.diff_size_input_listII[0][1] = config.token_to_count
print(f'Without classifier {model.foward_without_classification(config.diff_size_input_listII).argmax(-1)}')

class CNNCounter(nn.Module):
    def __init__(self, vocab_size, max_count, embed_dim=64, num_filters=128, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2)
        self.classifier = nn.Linear(num_filters, max_count + 1)  # +1 to include zero count
        
    def forward(self, x, temperature=1.0):
        embedded = self.embedding(x)
        x = embedded.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=2)  # Global average pooling across sequence
        logits = self.classifier(x) / temperature
        return logits

# Append this to the bottom of your file (after your existing code)
if __name__ == "__main__":
    # Initialize CNN model
    # Calculate maximum possible count based on your sequence length
    # For sequences of length 28 (excluding BOS/TMP/SEP), max_count = 25
    model_cnn = CNNCounter(
        vocab_size=7,  # BOS(0), TMP(1), SEP(2), 0(3), 1(4), 2(5), 3(6)
        max_count=25,
        embed_dim=64,
        num_filters=128,
        kernel_size=3
    ).to(config.device)

    def convert_labels(inputs):
        # Count the number of target_token (e.g., 4) in each input sequence after special tokens
        counts = [seq[3:].count(config.token_to_count) for seq in inputs]
        return torch.tensor(counts, dtype=torch.long).to(config.device)
    
    # Training setup with corrected loss
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss
    optimizer = optim.Adam(model_cnn.parameters(), lr=config.learning_rate)

    def cnn_train():
        model_cnn.train()
        for epoch in range(config.num_epochs):
            total_loss = 0.0
            for batch in config.train_loader:
                inputs, _ = batch  # Use inputs to compute labels, ignore original labels
                # Convert raw inputs to token indices (if not already done)
                labels = convert_labels(inputs).to(config.device)
                input_indices = model.encoder(inputs)  # Ensure this outputs token indices (Long tensor)
                
                # Compute correct labels based on input sequences
                
                outputs = model_cnn(input_indices)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss/len(config.train_loader):.4f}")
            loss_cnn_model.append(total_loss/len(config.train_loader))
            cnn_evaluate()
    
    def cnn_evaluate():
        model_cnn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in config.test_loader:
                inputs, labels = batch
                # Convert raw input tokens to indices
                true_counts = convert_labels(inputs).to(config.device)

                inputs = model.encoder(inputs)

                
                # Convert to actual counts
                
                outputs = model_cnn(inputs)
                predicted_counts = outputs.argmax(dim=-1)
                
                # Compare single count predictions
                correct += (predicted_counts == true_counts).sum().item()
                total += true_counts.numel()
        
        print(f"Validation Accuracy: {correct/total:.4f}\n")
    
    # Run training
    cnn_train()
    
    # Final evaluation
    print("\nFinal Test Results:")
    cnn_evaluate()

def evaluate_per_count(models_dict, m=100):
    """Evaluate models on m samples per count, returns accuracy per count"""
    original_model, cnn_model = models_dict['original'], models_dict['cnn']
    max_count = 25  # From your dataset setup
    results = {model_name: np.zeros(max_count+1) for model_name in models_dict.keys()}
    
    # Temporary dataset override for balanced evaluation
    class EvalDataset(Dataset):
        def __init__(self):
            self.data = []
            for count in range(max_count+1):
                for _ in range(m):
                    # Create sequence with exactly 'count' target tokens
                    seq = [config.bos, config.tmp, config.sep]
                    seq += [config.token_to_count]*count
                    seq += [str(np.random.choice(['0','2','3'])) 
                             for _ in range(25 - count)]  # 25 positions after special tokens
                    np.random.shuffle(seq[3:])  # Shuffle non-special tokens
                    self.data.append(seq[:28])  # Truncate to max_seq_len
                    
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], torch.tensor([0])  # Dummy label

    eval_loader = DataLoader(EvalDataset(), batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Evaluation loop
    with torch.no_grad():
        for batch in eval_loader:
            inputs, _ = batch
            
            # Original model prediction

            original_outputs = original_model(inputs).argmax(-1)
            original_counts = original_outputs[:, 1].cpu().numpy()  # TMP position
            
            # CNN model prediction
            encoded_inputs = model.encoder(inputs)
            cnn_outputs = cnn_model(encoded_inputs).argmax(-1).cpu().numpy()
            
            # True counts
            true_counts = [seq[3:].count(config.token_to_count) for seq in inputs]
            
            # Update results
            for tc, oc, cc in zip(true_counts, original_counts, cnn_outputs):
                results['original'][tc] += (oc == tc + 1)
                results['cnn'][tc] += (cc == tc)
    
    # Convert to accuracy percentages
    for model_name in results:
        results[model_name] = (results[model_name] / m) * 100
        
    return results

def plot_count_accuracy(results):
    """Plot accuracy per count for both models"""
    counts = np.arange(26)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 7))
    
    rects1 = ax.bar(counts - width/2, results['original'], width, label='Original Model')
    rects2 = ax.bar(counts + width/2, results['cnn'], width, label='CNN Model')
    
    ax.set_xlabel('True Count Value')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Count-wise Accuracy Comparison')
    ax.set_xticks(counts)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('count_accuracy_comparison.png')
    plt.show()

# Usage
models_to_evaluate = {
    'original': model,
    'cnn': model_cnn
}

accuracy_results = evaluate_per_count(models_to_evaluate, m=100)
# plot_count_accuracy(accuracy_results)


import json
count_freq = np.asarray(config.counts_dis).tolist()


train_count_dis = []
for batch in config.train_loader:
    input_values, _ = batch
    for el in input_values:
        train_count_dis.append(el.count(config.token_to_count))

test_count_dis = []
for batch in config.test_loader:
    input_values, _ = batch
    for el in input_values:
        test_count_dis.append(el.count(config.token_to_count))

# If your dictionaries may have arrays as values:
tok_his_clean = {
    key: val.tolist() if isinstance(val, np.ndarray) else val
    for key, val in token_history.items()
}
accuracy_res_clean = {
    key: val.tolist() if isinstance(val, np.ndarray) else val
    for key, val in accuracy_results.items()
}

data = {
    'count_freq': count_freq,
    'train_count_freq': train_count_dis,
    'test_count_freq': test_count_dis,
    'tok_his': tok_his_clean,
    'accuracy_res': accuracy_res_clean,
    'seed': config.used_seed,
    'loss_rasp_model': loss_rasp_model,
    'loss_cnn_model': loss_cnn_model
}

with open('run5_fixed_data_complete.json', 'w') as f:
    json.dump(data, f, indent=2)

