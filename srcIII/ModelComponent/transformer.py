import torch
import torch.nn as nn
from srcIII.ModelComponent import mlp
from srcIII.ModelComponent import attention

class Transformer_Module(nn.Module):
    def __init__(self, model_config, model_parameters, trainable=False, n_layers=1):
        super(Transformer_Module, self).__init__() 
        if not trainable:
            self.layers = []
            with torch.no_grad():
                for l in range(model_config.num_layers):
                    ATT_Layer = attention.ATT_Module(key_size = model_config.key_size, 
                                                    num_heads=model_config.num_heads,
                                                    query_data=model_parameters['layers'][l]['attn']['query'],
                                                    key_data=model_parameters['layers'][l]['attn']['key'],
                                                    value_data=model_parameters['layers'][l]['attn']['value'],
                                                    linear_data=model_parameters['layers'][l]['attn']['linear'],
                                                    trainable=False)
                    ATT_Layer.requires_grad_(False)
                    self.layers.append(ATT_Layer)

                    MLP_Layer = mlp.MLP_Module(activation_function=model_config.activation_function, 
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


