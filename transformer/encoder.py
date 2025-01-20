import torch
import torch.nn as nn

from attention import MultiHeadAttention
from mlp import FeedForward

class Encoder(nn.Module):
    """
    First section of the transformer, which encodes the input encoding into
    a full-context encoding for each token that will be passed to subsequent
    encoders or the decoder.
    """
    def __init__(self, d_model: int, num_attention_heads: int, d_ff: int, activation: nn.Module, layer_norm_epsilon: float):
        """
        Encoder initializer.

        Args:
            d_model - The embedding & hidden dimension of the transformer
            num_attention_heads -- The number of attention heads
            d_ff - The feed-forward layer's hidden dimension 
            activation - The activation function to use in the hidden linear layer at the end of each encoder/decoder block
            layer_norm_epsilon - The epsilon (numerical stability) to use for each LayerNorm layer
        """
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon

        self.mha = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.ff = FeedForward(self.d_model, self.d_ff, self.activation)

        # Must be distinct to learn independent distribution parameters (gamma, beta)
        self.layer_norm_1 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_2 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)

    def forward(self, x):
        """
        Pushes the input embedding through one full transformer encoder sequence.

        Output is compatible to be fed to either another encoder or the decoder.

        Args:
            x - The input embedding

        Returns:
            x - The full-context input embedding (via multi-head self-attention mechanism)
        """
        MHA = self.mha(x) # Multi-head Attention Mechanism
        x += MHA # Residual Connection
        x = self.layer_norm_1(x)

        FF = self.ff(x) # Feed-Forward Mechanism
        x += FF # Residual Connection
        x = self.layer_norm_2(x)

        return x