bos = "BOS"
sep = "SEP"
tmp = "TMP" 
null = "NLL"
custom_pad = null
max_rasp_len = 40

mlp_exactness = 1000000
vocab_req = {sep, tmp, null}
vocab_tokens = {'0', '1', '2', '3'}
vocab = vocab_tokens.union(vocab_req)
index_to_count = 1 # 1 for first token or -1 for last token
test_input_listI = [[bos,'1',sep,'1','1','2','2','1','2'], [bos,'1',sep,'1','1','2','2','1','2']]

learning_rate = 1e-3

# # full_dataset = False
logit_dataset = True
encoding_map = None


batch_size = 350
labels_samples = 7 #0 until 7
n_samples = 1500
train_split = 0.9
train_loader = None
test_loader = None
device = None

num_epochs = 100
temperature = 0.1

start_temp = 0.1
end_temp = 0.001

rasp_func = None
rasp_model = None
out_rasp = None

model_config = None
model_parameters = None
embedding_dim = None
output_dim = None
unembed_matrix = None
encoding_func = None

vocab_list = None
encoded_vocab= None
embedded_vocab = None

