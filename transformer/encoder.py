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
    def __init__(self, d_model: int, num_attention_heads: int, d_ff: int, activation: nn.Module = nn.ReLU(), layer_norm_epsilon: float = 1e-5):
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

        # Normalizes over the last dimension, d_model
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
        original_size = x.size()

        MHA = self.mha(x.clone(), x.clone(), x.clone()) # Multi-head Attention Mechanism
        assert MHA.size() == original_size
        x += MHA # Residual Connection
        x = self.layer_norm_1(x)
        assert x.size() == original_size

        FF = self.ff(x) # Feed-Forward Mechanism
        assert FF.size() == original_size
        x += FF # Residual Connection
        x = self.layer_norm_2(x)
        assert x.size() == original_size

        return x
    

if __name__ == '__main__':

    batch_size = 3
    seq = 5
    d_model = 8
    n_heads = 2
    d_ff = 4

    encoder = Encoder(d_model, n_heads, d_ff)

    E = torch.randn(batch_size, seq, d_model) 

    out: torch.Tensor = encoder(E)

    assert out.size() == E.size()