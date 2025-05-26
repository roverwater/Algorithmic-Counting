# Config
device = None

# RASP
vocab = {0, 1}
max_seq_len = 40
compiler_bos = 'BOS'
mlp_exactness = 1000000
input = [0, 1, 0, 1, 1, 0]
model = None
encoding_map = None
encoded_bos = None
out = None
unembed_matrix = None
model_params = None
model_config = None
output_dim = None

# Training params
learning_rate = 1e-3
num_epochs = 100
logit_dataset = True