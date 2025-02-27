import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism component of the transformer's encoder
    and decoder blocks.
    """
    def __init__(self, d_model: int, num_attention_heads: int, causal_mask = False):
        """
        MultiHeadAttention initializer.

        Args:
            d_model -- The embedding & hidden dimension of the transformer
            num_attention_heads -- The number of attention heads
            causal_mask -- Whether or not to allow later words in a sequence
            to attend to earlier words.
        """
        super().__init__()
        self.d_model = d_model

        self.num_attention_heads = num_attention_heads # 'h' in "Attention Is All You Need"
        self.d_k = self.d_v = self.d_model / self.num_attention_heads

        self.causal_mask = causal_mask

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model) 

        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, in_Q: torch.Tensor, in_K: torch.Tensor, in_V: torch.Tensor, row_pad_mask: torch.Tensor, col_pad_mask: torch.Tensor):
        """
        Pushes the input embedding E through a multi-head attention mechanism,
        returning a tensor of the same dimensions. The embeddings in the output
        now all contain context about the surrounding words, in addition to their
        own embedding and positional encoding.

        Args:
            in_Q - The Q input embedding to be passed through the attention mechanism of
            shape (batch_size, seq, d_model). 
            in_K - The K input embedding to be passed through the attention mechanism of
            shape (batch_size, seq, d_model). 
            in_V - The V input embedding to be passed through the attention mechanism of
            shape (batch_size, seq, d_model). 
            row_pad_mask - Indicator of row padding locations to mask (so as to not contribute to attention).
            col_pad_mask - Indicator of column padding locations to mask (so as to not contribute to attention).

        Returns:
            MHA - The result of a Multi-head Attention mechanism on E of shape
            (batch_size, seq, d_model).
        """
        # if torch.allclose(in_Q, in_K) and torch.allclose(in_K, in_V): # If not in cross-attention layer...
        #     assert in_Q.size() == in_K.size() == in_V.size()
        # else:
        #     assert not torch.allclose(in_Q, in_V) and torch.allclose(in_K, in_V) # Verify cross-attention property

        in_batch_size, in_seq_len_Q, _ = in_Q.size()
        _, in_seq_len_K, _ = in_K.size()

        Q, K, V = self.W_Q(in_Q), self.W_K(in_K), self.W_V(in_V) # all (batch_size, seq, d_model)
        K_T = torch.transpose(K, 1, 2) 

        # Q, V shape (batch_size, seq, d_model)
        # K_T shape (batch_size, d_model, seq)

        Q_h = torch.chunk(Q, self.num_attention_heads, dim = 2) # list of h tensors of shape (batch_size, seq, d_k = d_model / h)
        K_T_h = torch.chunk(K_T, self.num_attention_heads, dim = 1) # list of h tensors of shape (batch_size, d_k = d_model / h, seq)
        V_h = torch.chunk(V, self.num_attention_heads, dim = 2) # list of h tensors of shape (batch_size, seq, d_k = d_model / h)

        assert len(Q_h) == len(K_T_h) == len(V_h) == self.num_attention_heads

        # Now, we WANT...
        # Q_h_stack, V_h_stack shapes of (batch_size, h, seq, d_k)
        # K_T_h_stack shape of (batch_size, h, d_k, seq)

        Q_h_stack = torch.stack(Q_h, dim = 1)
        K_T_h_stack = torch.stack(K_T_h, dim = 1)
        V_h_stack = torch.stack(V_h, dim = 1)

        assert Q_h_stack.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, self.d_k)
        assert K_T_h_stack.transpose(2, 3).size() == V_h_stack.size() == (in_batch_size, self.num_attention_heads, in_seq_len_K, self.d_k)

        # Within each batch, between the matching batch index Q = Q_h_stack[i] and K = K_T_h_stack[i],
        # we will perform h (Q @ K_T) matrix multiplications, producing a new set (of len h) of tensors
        # of shape (batch_size, h, seq, seq).

        # Note: matmul treats leading dimensions as independent of matrix product; batch_size * h matrix multiplications will occur
        Q_KT = torch.matmul(Q_h_stack, K_T_h_stack)

        assert Q_KT.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, in_seq_len_K)

        Q_KT_scaled = Q_KT / math.sqrt(self.d_k)

        # CAUSAL MASK
        # If enabled, ensure that seq_i in Q cannot see seq_j in K when j > i.
        # In other words, do not let earlier words (in Q) attend to later words (in K).
        if self.causal_mask:
            assert in_seq_len_Q == in_seq_len_K # Should not enter this block during cross-attention
            neg_inf_matrix = torch.full_like(Q_KT_scaled[0][0], float('-inf'))
            neg_inf_mask = neg_inf_matrix.triu(diagonal = 1) # if called on 4D tensor, triu applies batch*h times to each (seq, seq) dimension, only operates on 2D matrix
            Q_KT_scaled += neg_inf_mask # if 2D tensor, (seq, seq) will broadcast into (batch_size, h, seq, seq) to mask each (seq, seq)

        # PAD MASK
        assert row_pad_mask.ndim == col_pad_mask.ndim == 2
        row_batch_seq_lengths = row_pad_mask.sum(dim = 1) # length of each sequence (excluding PAD_TOKEN_IDX, see top of Transformer forward for creation)
        col_batch_seq_lengths = col_pad_mask.sum(dim = 1)

        row_seq_indices = torch.arange(in_seq_len_Q)
        col_seq_indices = torch.arange(in_seq_len_K)

        row_non_padding = row_seq_indices.unsqueeze(0) < row_batch_seq_lengths.unsqueeze(1) # shape = (batch, in_seq_len_Q)
        col_non_padding = col_seq_indices.unsqueeze(0) < col_batch_seq_lengths.unsqueeze(1) # shape = (batch, in_seq_len_K)

        non_padding = row_non_padding.unsqueeze(2) & col_non_padding.unsqueeze(1) # shape = (batch, in_seq_len_Q, in_seq_len_K)
        non_padding_h = non_padding.unsqueeze(1) # shape = (batch, 1, in_seq_len_Q, in_seq_len_K)

        assert non_padding_h.size() == (in_batch_size, 1, in_seq_len_Q, in_seq_len_K)

        Q_KT_scaled_masked = Q_KT_scaled.masked_fill(~non_padding_h, float('-inf'))

        assert Q_KT_scaled_masked.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, in_seq_len_K)

        # softmax should be applied row-wise on the (t_seq, s_seq) internal matrix. Thus, batch_size * h * t_seq softmaxes will occur.
        # dim = -1 used to pull out and softmax each row (by collapsing in the column/last dimension).
        importances = torch.softmax(Q_KT_scaled_masked, dim = -1)

        assert importances.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, in_seq_len_K)

        importances = importances.nan_to_num() # Fill in bottom rows nan output from softmax with 0

        # Verify that softmax was correctly applied row_wise to produce batch_size*h seq-length vectors of ones
        # assert torch.allclose(torch.ones(in_batch_size, self.num_attention_heads, in_seq), importances.sum(dim = -1))

        # Recall that importances is of shape (batch_size, h, seq, seq)
        # Recall that V_h_stack is of shape (batch_size, h, seq, d_v = d_k)
        # To compute each head's final attention, the last two dimensions must
        # be matrix multiplied by each other, again batch_size * h times to
        # produce a tensor of size (batch_size, h, seq, d_v).

        h_attention = torch.matmul(importances, V_h_stack)

        assert h_attention.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, self.d_v)

        # The final linear W_O maps between d_model dimensions. Thus, we must
        # merge the h final h_attention's horizontally back into a shape
        # of (batch_size, seq, d_model).

        H: torch.Tensor = h_attention.permute(0, 2, 1, 3).contiguous().view(in_batch_size, in_seq_len_Q, -1) # contiguous() call requires because permute breaks data continuity.

        assert H.size() == in_Q.size()
        
        MHA = self.W_O(H)

        assert MHA.size() == in_Q.size()

        return MHA
    