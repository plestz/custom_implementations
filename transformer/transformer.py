import torch
import torch.nn as nn

from utils import pad_batch_to_longest

class Transformer(nn.Module):
    """
    """
    def __init__(self, 
                 d_model = 512, 
                 num_attention_heads = 8, 
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 dim_feedforward = 2048,
                 activation = nn.ReLU,
                 layer_norm_epsilon = 1e-5):
        """
        """
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon

    def forward(self, source: list[torch.Tensor], target: torch.Tensor):
        """

        Args:
            source -- Encoder input sequence. A 1D list of 2D tensor of shape (d_embedding, n_tokens_i). Each list element is batch example i.
            target -- Decoder input sequence. A 1D list of 2D tensor of shape (d_embedding, n_tokens_i). Each list element is batch example i.
        """
        batch: torch.Tensor = pad_batch_to_longest(source, pad_value = 0)
    
if __name__ == '__main__':
    t = Transformer(d_model = 4, num_attention_heads = 4, num_encoder_layers = 2, num_decoder_layers = 2, dim_feedforward = 64)