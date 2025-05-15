import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.mlp import FeedForward

class Encoder(nn.Module):
    """
    First section of the transformer, which encodes the input encoding into
    a full-context encoding for each token that will be passed to subsequent
    encoders or the decoder.
    """
    def __init__(self, d_model: int, num_attention_heads: int, d_ff: int, dropout: float, activation: nn.Module = nn.ReLU(), layer_norm_epsilon: float = 1e-5, use_pre_lnorm: bool = False):
        """
        Encoder initializer.

        Args:
            d_model - The embedding & hidden dimension of the transformer
            num_attention_heads -- The number of attention heads
            d_ff - The feed-forward layer's hidden dimension 
            dropout - The dropout percentage for the network
            activation - The activation function to use in the hidden linear layer at the end of each encoder/decoder block
            layer_norm_epsilon - The epsilon (numerical stability) to use for each LayerNorm layer
            use_pre_lnorm - A flag to determine whether to use pre-norm (if set to true) or post-norm (otherwise)
        """
        super().__init__()
        self.d_model: int = d_model
        self.num_attention_heads: int = num_attention_heads
        self.d_ff: int = d_ff
        self.dropout: float = dropout
        self.activation: nn.Module = activation
        self.layer_norm_epsilon: float = layer_norm_epsilon
        self.use_pre_lnorm = use_pre_lnorm

        self.mha = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.ff = FeedForward(self.d_model, self.d_ff, self.activation, self.dropout)

        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

        # Normalizes over the last dimension, d_model
        # Must be distinct to learn independent distribution parameters (gamma, beta)
        self.layer_norm_1 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_2 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)

    def forward(self, x, source_pad_mask: torch.Tensor):
        """
        Pushes the input embedding through one full transformer encoder sequence.

        Output is compatible to be fed to either another encoder or the decoder.

        Args:
            x - The input embedding
            source_pad_mask - Indicator of padding locations to mask (so as to not contribute to attention)

        Returns:
            x - The full-context input embedding (via multi-head self-attention mechanism)
        """
        original_size = x.size()

        if self.use_pre_lnorm:
            x_lnorm = self.layer_norm_1(x)
            MHA = self.mha(x_lnorm, x_lnorm, x_lnorm, source_pad_mask, source_pad_mask)
            assert MHA.size() == original_size
            x = x + self.dropout_1(MHA)
            assert x.size() == original_size

            x_lnorm = self.layer_norm_2(x)
            FF = self.ff(x_lnorm)
            assert FF.size() == original_size
            x = x + self.dropout_2(FF)
            assert x.size() == original_size
            
        else: # use post-lnorm
            MHA = self.mha(x, x, x, source_pad_mask, source_pad_mask) # Multi-head Attention Mechanism
            assert MHA.size() == original_size
            x = x + self.dropout_1(MHA) # Residual Connection
            x = self.layer_norm_1(x)
            assert x.size() == original_size

            FF = self.ff(x) # Feed-Forward Mechanism
            assert FF.size() == original_size
            x = x + self.dropout_2(FF) # Residual Connection
            x = self.layer_norm_2(x)
            assert x.size() == original_size

        return x