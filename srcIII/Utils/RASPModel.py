import sys
sys.path.append('/home/ruov/projects/AlgorithmicCounting/')

import jax
import torch
from tracr.rasp import rasp
from tracr.compiler import compiling
from srcIII import config
from srcIII.Utils import utils


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

        config.encoding_map = config.rasp_model.input_encoder.encoding_map   

def encoding_func(x):
    encoded_samples = []
    print(config.rasp_model.input_encoder.encoding_map)
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
    config.model_parameters = utils.extract_weights(config.rasp_model.params)

    config.unembed_matrix = config.out_rasp.unembed_matrix
    config.encoding_func = encoding_func
    config.embedding_dim = config.unembed_matrix.shape[0]
    config.output_dim = config.unembed_matrix.shape[1]

    config.vocab_list = [[config.bos] + list(config.vocab)]
    config.encoded_vocab = config.encoding_func(config.vocab_list)

    return 
