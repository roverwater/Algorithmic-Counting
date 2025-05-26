import sys
sys.path.append('/home/ruov/projects/AlgorithmicCounting/')

import torch.nn as nn
from srcIII import config
from srcIII.ModelComponent import embedding
from srcIII.ModelComponent import transformer
from srcIII.ModelComponent import unembedding
from srcIII.Classifier import image_transformer
from srcIII.Classifier import embedder_classifier
from srcIII.Classifier import dino_transformer



class Model_Image(nn.Module):
    def __init__(self, model_config, model_parameters, unembedding_data, encoding_func, encoded_vocab):
        super(Model_Image, self).__init__()

        # self.image_transformer = image_transformer.Image_Transformer()

        vocab_size = model_parameters['embeddings']['token_embed']['embeddings'].shape[0] - 5

        # Initialize DINOv2 transformer
        self.image_transformer = dino_transformer.DINOv2_Image_Transformer(
            vocab_size=vocab_size
        ).to(config.device)

        self.embedding_dim = unembedding_data.shape[0]

        self.encoder = encoding_func

        self.embedder = embedding.Embedding_Module(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings'],
                                                                  trainable=False).to(config.device)

        self.classifier = embedder_classifier.Classifier(pos_embed_data=model_parameters['embeddings']['pos_embed']['embeddings'], 
                                                                  token_embed_data=model_parameters['embeddings']['token_embed']['embeddings']).to(config.device)
        
        self.transformer = transformer.Transformer_Module(model_config=model_config,
                                                                         model_parameters=model_parameters,
                                                                         trainable=False,
                                                                         )
        
        self.unembedder = unembedding.Unembedding_Module(unembedding_data=unembedding_data,
                                                                        use_unembed_argmax=False,
                                                                        trainable=False)
        self.embedder.requires_grad_(False)
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
