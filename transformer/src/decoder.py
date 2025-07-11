import torch
import torch.nn as nn
from src.attention import MultiHeadAttention
from src.mlp import FeedForward

class Decoder(nn.Module):
    """
    Second section of the transformer, which performs causal
    MHA on the decoder input, cross-attention on these embeddings with the
    encoder's output K and V, and feeds this forward through an MLP to obtain
    the final contextual embeddings.
    """
    def __init__(self, d_model: int, num_attention_heads: int, d_ff: int, dropout: float, activation: nn.Module = nn.ReLU(), layer_norm_epsilon: float = 1e-5, use_pre_lnorm: bool = False):
        """
        Decoder initializer.

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

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_pre_lnorm = use_pre_lnorm

        self.masked_mha = MultiHeadAttention(self.d_model, self.num_attention_heads, enable_causal_mask = True)
        self.mha = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.ff = FeedForward(self.d_model, self.d_ff, self.activation, self.dropout)

        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)
        self.dropout_3 = nn.Dropout(self.dropout)

        # Normalizes over the last dimension, d_model
        # Must be distinct to learn independent distribution parameters (gamma, beta)
        self.layer_norm_1 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_2 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.layer_norm_3 = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)

    def forward(self, x, encoder_K: torch.Tensor, encoder_V: torch.Tensor, target_pad_mask: torch.Tensor, source_pad_mask: torch.Tensor):
        """
        Pushes the output embedding through one full transformer deocder sequence.

        Output is compatible to be fed to either another decoder or towards the output probability layer.

        Note: Assumes decoder-only origin to disable cross-attention if any of encoder_K, encoder_V, source_pad_mask are None.

        Args:
            x - The decoder input of shape (batch_size, seq, d_model)
            encoder_K - The K tensor output by the final encoder block.
            encoder_V - The V tensor output by the final encoder block.
            target_pad_mask - Indicator of target padding locations to mask (so as to not contribute to attention)
            source_pad_mask - Indicator of source padding locations to mask (so as to not contribute to attention)

        Returns:
            x - The full-context decoder embedding (via (causal) multi-head self-attention and [optional] cross-attention mechanisms) of
            shape (batch_size, seq, d_model). Input x.size() = Output x.size()
        """
        original_size = x.size()

        # For decoder-only functionality
        disable_flag_decoder_only = not (encoder_K is None or encoder_V is None or source_pad_mask is None)

        if self.use_pre_lnorm:
            # CAUSAL SELF-ATTENTION
            x_lnorm = self.layer_norm_1(x)
            MASKED_MHA = self.masked_mha(x_lnorm, x_lnorm, x_lnorm, target_pad_mask, target_pad_mask)
            assert MASKED_MHA.size() == original_size
            x = x + self.dropout_1(MASKED_MHA)
            assert x.size() == original_size

            # CROSS-ATTENTION
            if disable_flag_decoder_only:
                x_lnorm = self.layer_norm_2(x)
                MHA = self.mha(x_lnorm, encoder_K, encoder_V, target_pad_mask, source_pad_mask)
                assert MHA.size() == original_size
                x = x + self.dropout_2(MHA)
                assert x.size() == original_size

            # FFN
            x_lnorm = self.layer_norm_3(x)
            FF = self.ff(x_lnorm)
            assert FF.size() == original_size
            x = x + self.dropout_3(FF)
            assert x.size() == original_size
        
        else: # use post-lnorm
            # CAUSAL SELF-ATTENTION
            MASKED_MHA = self.masked_mha(x, x, x, target_pad_mask, target_pad_mask)
            assert MASKED_MHA.size() == original_size
            x = x + self.dropout_1(MASKED_MHA)
            x = self.layer_norm_1(x)
            assert x.size() == original_size

            # CROSS-ATTENTION
            if disable_flag_decoder_only:
                MHA = self.mha(x, encoder_K, encoder_V, target_pad_mask, source_pad_mask)
                assert MHA.size() == original_size
                x = x + self.dropout_2(MHA)
                x = self.layer_norm_2(x)
                assert x.size() == original_size

            # FFN
            FF = self.ff(x)
            assert FF.size() == original_size
            x = x + self.dropout_3(FF)
            x = self.layer_norm_3(x)
            assert x.size() == original_size

        return x