import torch
import torch.nn as nn
from attention import MultiHeadAttention
from mlp import FeedForward

class Decoder(nn.Module):
    """
    Second section of the transformer, which performs causal
    MHA on the decoder input, cross-attention on these embeddings with the
    encoder's output K and V, and feeds this forward through an MLP to obtain
    the final contextual embeddings.
    """
    def __init__(self, d_model: int, num_attention_heads: int, d_ff: int, encoder_K: torch.Tensor, encoder_V: torch.Tensor, activation: nn.Module = nn.ReLU(), layer_norm_epsilon: float = 1e-5):
        """
        Decoder initializer.

        Args:
            d_model - The embedding & hidden dimension of the transformer
            num_attention_heads -- The number of attention heads
            d_ff - The feed-forward layer's hidden dimension 
            encoder_K - The K tensor output by the final encoder block.
            encoder_V - The V tensor output by the final encoder block.
            activation - The activation function to use in the hidden linear layer at the end of each encoder/decoder block
            layer_norm_epsilon - The epsilon (numerical stability) to use for each LayerNorm layer
        """
        super().__init__()

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon

        self.encoder_K = encoder_K
        self.encoder_V = encoder_V

        self.masked_mha = MultiHeadAttention(self.d_model, self.num_attention_heads, causal_mask = True)
        self.mha = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.ff = FeedForward(self.d_model, self.d_ff, self.activation)

        # Normalizes over the last dimension, d_model
        # Must be distinct to learn independent distribution parameters (gamma, beta)
        self.layer_norm_1 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_2 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_3 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)

    def forward(self, x):
        """
        Pushes the output embedding through one full transformer deocder sequence.

        Output is compatible to be fed to either another decoder or towards the output probability layer.

        Args:
            x - The decoder input

        Returns:
            x - The full-context decoder embedding (via (causal) multi-head self-attention and cross-attention mechanisms)
        """
        original_size = x.size()

        MASKED_MHA = self.masked_mha(x.clone(), x.clone(), x.clone())
        assert MASKED_MHA.size() == original_size
        x += MASKED_MHA
        x = self.layer_norm_1(x)
        assert x.size() == original_size

        MHA = self.mha(x.clone(), self.encoder_K, self.encoder_V)
        assert MHA.size() == original_size
        x += MHA
        x = self.layer_norm_2(x)
        assert x.size() == original_size

        FF = self.ff(x)
        assert FF.size() == original_size
        x += FF
        x = self.layer_norm_3(x)
        assert x.size() == original_size

        return x

if __name__ == '__main__':

    batch_size = 3
    seq = 5
    d_model = 8
    n_heads = 2
    d_ff = 4

    E = torch.randn(batch_size, seq, d_model) 

    decoder = Decoder(d_model, n_heads, d_ff, E.clone(), E.clone())

    out: torch.Tensor = decoder(E)

    assert out.size() == E.size()