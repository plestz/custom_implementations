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
    def __init__(self, d_model: int, num_attention_heads: int, d_ff: int, activation: nn.Module = nn.ReLU(), layer_norm_epsilon: float = 1e-5):
        """
        Decoder initializer.

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

        self.masked_mha = MultiHeadAttention(self.d_model, self.num_attention_heads, enable_causal_mask = True)
        self.mha = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.ff = FeedForward(self.d_model, self.d_ff, self.activation)

        # Normalizes over the last dimension, d_model
        # Must be distinct to learn independent distribution parameters (gamma, beta)
        self.layer_norm_1 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_2 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_3 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)

    def forward(self, x, encoder_K: torch.Tensor, encoder_V: torch.Tensor, target_pad_mask: torch.Tensor, source_pad_mask: torch.Tensor):
        """
        Pushes the output embedding through one full transformer deocder sequence.

        Output is compatible to be fed to either another decoder or towards the output probability layer.

        Args:
            x - The decoder input of shape (batch_size, seq, d_model)
            encoder_K - The K tensor output by the final encoder block.
            encoder_V - The V tensor output by the final encoder block.
            target_pad_mask - Indicator of target padding locations to mask (so as to not contribute to attention)
            source_pad_mask - Indicator of source padding locations to mask (so as to not contribute to attention)

        Returns:
            x - The full-context decoder embedding (via (causal) multi-head self-attention and cross-attention mechanisms) of
            shape (batch_size, seq, d_model). Input x.size() = Output x.size()
        """
        original_size = x.size()

        # CAUSAL SELF-ATTENTION
        MASKED_MHA = self.masked_mha(x.clone(), x.clone(), x.clone(), target_pad_mask, target_pad_mask)
        assert MASKED_MHA.size() == original_size
        x = x + MASKED_MHA
        x = self.layer_norm_1(x)
        assert x.size() == original_size

        # CROSS-ATTENTION
        MHA = self.mha(x.clone(), encoder_K, encoder_V, target_pad_mask, source_pad_mask)
        assert MHA.size() == original_size
        x = x + MHA
        x = self.layer_norm_2(x)
        assert x.size() == original_size

        FF = self.ff(x)
        assert FF.size() == original_size
        x = x + FF
        x = self.layer_norm_3(x)
        assert x.size() == original_size

        return x