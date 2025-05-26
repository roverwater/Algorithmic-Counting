import sys
sys.path.append('/home/ruov/projects/AlgorithmicCounting/')

import jax
import torch
from srcIV import config
from tracr.rasp import rasp
from srcIV.Utils import utils
from srcIV.Model import blank_model
from tracr.compiler import compiling

jax.config.update("jax_default_matmul_precision", "highest")

def make_count(sop, token):
    return rasp.SelectorWidth(rasp.Select(sop, sop, lambda k, q: k == token)).named(f"count_{token}")

def compile_model():
    model = compiling.compile_rasp_to_model(
        make_count(rasp.tokens, 1),
        vocab=config.vocab,
        max_seq_len=config.max_seq_len,
        compiler_bos=config.compiler_bos,
        mlp_exactness=config.mlp_exactness
    )
    return model

def compile():
    config.model = compile_model()
    config.encoding_map = config.model.input_encoder.encoding_map
    print(config.encoding_map)
    config.encoded_bos = config.encoding_map[config.compiler_bos]
    config.out = config.model.apply([config.compiler_bos] + config.input)
    config.unembed_matrix = config.out.unembed_matrix
    config.output_dim = config.unembed_matrix.shape[1]
    config.model_params = utils.extract_weights(config.model.params)
    config.model_config = config.model.model_config
    config.model_config.activation_function = torch.nn.ReLU()

    # pytorch_model = blank_model.Model_Blank(model_config=config.model_config,
    #                                         model_parameters=config.model_params,
    #                                          unembedding_data=config.unembed_matrix)
    # input_pytorch = torch.tensor([config.encoded_bos] + config.input)
    # out_pytorch = pytorch_model(input_pytorch)
    # print(config.out.decoded)
    # print(out_pytorch.argmax(-1))

    
if __name__ == "__main__":
    compile()

    
    # print(model_params['layers'])



# def CompileRaspModel():
#     model_class = Models()
#     config.rasp_func = model_class.count_agnostic_first()
#     model_class.compile_model()
#     config.out_rasp = config.rasp_model.apply(config.test_input_listI[0])
#     print(f"Count RASP token: '{str(config.test_input_listI[0][config.index_to_count])}' expected: {str(config.test_input_listI[0].count(config.test_input_listI[0][config.index_to_count]))}, computed: {str(config.out_rasp.decoded[-1])}, raw out: {config.out_rasp.decoded}")

#     config.model_config = config.rasp_model.model_config
#     config.model_config.activation_function = torch.nn.ReLU()
#     config.model_parameters = utils.extract_weights(config.rasp_model.params)

#     config.unembed_matrix = config.out_rasp.unembed_matrix
#     config.encoding_func = encoding_func
#     config.embedding_dim = config.unembed_matrix.shape[0]
#     config.output_dim = config.unembed_matrix.shape[1]

#     config.vocab_list = [[config.bos] + list(config.vocab)]
#     config.encoded_vocab = config.encoding_func(config.vocab_list)

#     return 
