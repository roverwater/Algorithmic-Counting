import sys
sys.path.append('/home/ruov/projects/AlgorithmicCounting/')

import jax
import torch
from tracr.rasp import rasp
from srcIII.Utils import utils
from srcIII.Model import blank_model
from tracr.compiler import compiling

jax.config.update("jax_default_matmul_precision", "highest")

def make_count(sop, token):
    return rasp.SelectorWidth(rasp.Select(sop, sop, lambda k, q: k == token)).named(f"count_{token}")

def compile_model():
    model = compiling.compile_rasp_to_model(
        make_count(rasp.tokens, 1),
        vocab={0, 1},
        max_seq_len=30,
        compiler_bos='BOS',
        mlp_exactness=1000000
    )
    return model
    
if __name__ == "__main__":
    model = compile_model()
    encoding_map = model.input_encoder.encoding_map
    input = [0, 1, 0, 1, 1, 0]
    input_pytorch = torch.tensor([encoding_map['BOS']] + input)
    out = model.apply(['BOS'] + input)
    print(out.decoded)
    unembed_matrix = out.unembed_matrix
    model_params = utils.extract_weights(model.params)
    model_config = model.model_config
    model_config.activation_function = torch.nn.ReLU()

    pytorch_model = blank_model.Model_Blank(model_config=model_config,
                                            model_parameters=model_params,
                                             unembedding_data=unembed_matrix)
    out_pytorch = pytorch_model(input_pytorch)
    print(out_pytorch.argmax(-1))
    
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
