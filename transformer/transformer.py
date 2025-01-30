import torch
import torch.nn as nn

import numpy as np

from encoder import Encoder
from decoder import Decoder
from utils import pad_batch_to_longest

class Transformer(nn.Module):
    """
    My custom implementation of a Transformer, based off of PyTorch's.
    """
    def __init__(self, 
                 d_model = 512, 
                 num_attention_heads = 8, 
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 dim_feedforward = 2048,
                 activation = nn.ReLU(),
                 layer_norm_epsilon = 1e-5,
                 max_context_window = 1024):
        """
        Transformer intitializer.

        Args:
            d_model - The embedding & hidden dimension of the transformer
            num_attention_heads - The number of attention heads
            num_encoder_layers - The number of sequential encoder blocks in the network
            num_decoder_layers - The number of sequential decoder blocks in the network
            dim_feedforward - The feed-forward layer's hidden dimension 
            activation - The activation function to use in the hidden linear layer at the end of each encoder/decoder block
            layer_norm_epsilon - The epsilon (numerical stability) to use for each LayerNorm layer
            max_context_window - The maximum context window that this transformer will be used to process
        """
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f'd_model must be even, but was {d_model}.')
        
        if max_context_window % 2 != 0:
            raise ValueError(f'max_context_window must be even, but was {max_context_window}.')

        self.d_model: int = d_model
        self.num_attention_heads: int = num_attention_heads
        self.num_encoder_layers: int = num_encoder_layers
        self.num_decoder_layers: int = num_decoder_layers
        self.dim_feedforward: int = dim_feedforward
        self.activation: nn.Module = activation
        self.layer_norm_epsilon: float = layer_norm_epsilon

        # 1024 = GPT2
        self.max_context_window = max_context_window

        self.positional_encodings = self.get_all_positional_encodings().float()

    def forward(self, source: list[torch.Tensor], target: torch.Tensor):
        """
        Performs one full forward pass through the transformer. 
        
        More detail coming soon.

        Args:
            source -- Encoder input batch. A 1D list of 2D tensor of shape (seq, d_model). Each list element is batch example i.
            target -- Decoder input batch. A 1D list of 2D tensor of shape (seq, d_model). Each list element is batch example i.
        """
        # Validate that d_embedding = d_model
        assert source[0].size(1) == self.d_model

        input_embedding, max_input_seq = pad_batch_to_longest(source, pad_value = 0)
        output_embedding, max_output_seq = pad_batch_to_longest(target, pad_value = 0)

        # Encoders (Sequential Processing)
        encoders = nn.ModuleList([Encoder(self.d_model, self.num_attention_heads, self.dim_feedforward, self.activation, self.layer_norm_epsilon) for _ in range(self.num_encoder_layers)])

        encoder_input = input_embedding + self.positional_encodings[:max_input_seq].unsqueeze(0)

        encoder_output = encoder_input
        for i in range(self.num_encoder_layers):
            encoder_output = encoders[i](encoder_output)

        encoder_K, encoder_V = encoder_output.clone(), encoder_output.clone()

        # Decoders (Sequential Processing)
        decoders = nn.ModuleList([Decoder(self.d_model, self.num_attention_heads, self.dim_feedforward, encoder_K, encoder_V, self.activation, self.layer_norm_epsilon) for _ in range(self.num_decoder_layers)])

        decoder_input = output_embedding + self.positional_encodings[:max_output_seq].unsqueeze(0)

        decoder_output = decoder_input
        for i in range(self.num_decoder_layers):
            decoder_output = decoders[i](decoder_output)

        # out (linear + softmax)

        return decoder_output

    def get_all_positional_encodings(self) -> torch.Tensor:
        """
        Produces all possible positional encodings according to the maximum
        context window and d_model, to be stored once at transformer creation.

        Returns:
            positional_encodings - All possible positional encodings in the shape
            of (seq, d_model)
        """
        positional_encodings = np.empty((self.max_context_window, self.d_model))

        positions = np.arange(self.max_context_window).reshape(-1, 1)
        dimensions = np.arange(self.d_model // 2)
        scaling_factor = np.power(10000, 2 * dimensions / self.d_model).reshape(1, -1)

        # Note: np.divide broadcasts two arrays of type [N, 1], [1, M] to [N, M]
        raw_encodings = np.divide(positions, scaling_factor)
        sin_encodings = np.sin(raw_encodings)
        cos_encodings = np.cos(raw_encodings)

        positional_encodings[np.ix_(positions.flatten(), dimensions * 2)] = sin_encodings
        positional_encodings[np.ix_(positions.flatten(), dimensions * 2 + 1)] = cos_encodings

        return torch.from_numpy(positional_encodings)
    
if __name__ == '__main__':

    batch_size = 3
    seq = 5
    d_model = 8
    n_heads = 2
    d_ff = 4

    E = torch.randn(batch_size, seq, d_model) 
    T = torch.randn(batch_size, seq, d_model) 

    transformer = Transformer(d_model = d_model, num_attention_heads = n_heads, num_encoder_layers = 6, num_decoder_layers = 6, dim_feedforward = d_ff, max_context_window = 20)

    out = transformer(E, T)

    assert out.size() == E.size()