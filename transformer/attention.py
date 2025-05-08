import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism component of the transformer's encoder
    and decoder blocks.
    """
    def __init__(self, d_model: int, num_attention_heads: int, enable_causal_mask = False):
        """
        MultiHeadAttention initializer.

        Args:
            d_model -- The embedding & hidden dimension of the transformer
            num_attention_heads -- The number of attention heads
            enable_causal_mask -- Whether or not to allow later words in a sequence
            to attend to earlier words.
        """
        super().__init__()

        self.d_model = d_model

        self.num_attention_heads = num_attention_heads # 'h' in "Attention Is All You Need"
        self.d_k = self.d_v = self.d_model // self.num_attention_heads

        self.enable_causal_mask = enable_causal_mask

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model) 

        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, in_Q: torch.Tensor, in_K: torch.Tensor, in_V: torch.Tensor, Q_pad_mask: torch.Tensor, K_pad_mask: torch.Tensor):
        """
        Pushes the input embedding E through a multi-head attention mechanism,
        returning a tensor of the same dimensions. The embeddings in the output
        now all contain context about the surrounding words, in addition to their
        own embedding and positional encoding.

        If enable_causal_mask is set to True, then "surrounding words" means only
        each word's left-context.

        Args:
            in_Q - The Q input embedding to be passed through the attention mechanism of
            shape (batch_size, seq, d_model). 
            in_K - The K input embedding to be passed through the attention mechanism of
            shape (batch_size, seq, d_model). 
            in_V - The V input embedding to be passed through the attention mechanism of
            shape (batch_size, seq, d_model). 
            Q_pad_mask - Indicator of row padding locations to mask (so as to not contribute to attention).
            K_pad_mask - Indicator of column padding locations to mask (so as to not contribute to attention).

        Returns:
            MHA - The result of a Multi-head Attention mechanism on E of shape
            (batch_size, seq, d_model).
        """
        in_batch_size, in_seq_len_Q, _ = in_Q.size()
        _, in_seq_len_K, _ = in_K.size()

        Q, K, V = self.W_Q(in_Q), self.W_K(in_K), self.W_V(in_V) # all (batch_size, seq, d_model)
        K_T = torch.transpose(K, 1, 2) 

        assert Q.size() == V.size() == (in_batch_size, in_seq_len_Q, self.d_model)
        assert K_T.size() == (in_batch_size, self.d_model, in_seq_len_K)

        # Note: In XTransformerClass, we ensured that d_model % h = 0. So, chunks will be of even sizes. Note that torch.chunk does not guarantee this.
        Q_h = torch.chunk(Q, self.num_attention_heads, dim = 2) # tuple of h tensors of shape (batch_size, seq, d_k = d_model // h)
        K_T_h = torch.chunk(K_T, self.num_attention_heads, dim = 1) # tuple of h tensors of shape (batch_size, d_k = d_model // h, seq)
        V_h = torch.chunk(V, self.num_attention_heads, dim = 2) # tuple of h tensors of shape (batch_size, seq, d_k = d_model // h)

        # Now, we WANT...
        # Q_h_stack, V_h_stack shapes of (batch_size, h, seq, d_k)
        # K_T_h_stack shape of (batch_size, h, d_k, seq)

        Q_h_stack = torch.stack(Q_h, dim = 1)
        K_T_h_stack = torch.stack(K_T_h, dim = 1)
        V_h_stack = torch.stack(V_h, dim = 1)

        assert Q_h_stack.size() == V_h_stack.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, self.d_k)
        assert K_T_h_stack.size() == (in_batch_size, self.num_attention_heads, self.d_k, in_seq_len_K)

        # Within each batch element, between the matching batch index Q = Q_h_stack[i] and K = K_T_h_stack[i],
        # we will perform h (Q @ K_T) matrix multiplications, producing a new set (of len h) of tensors
        # of shape (batch_size, h, seq, seq).

        # Note: matmul (@) treats leading dimensions as independent of matrix product; batch_size * h matrix multiplications will occur
        Q_KT = Q_h_stack @ K_T_h_stack

        assert Q_KT.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, in_seq_len_K)

        Q_KT_scaled = Q_KT / math.sqrt(self.d_k)

        # CAUSAL MASK
        # If enabled, ensure that seq_i in Q cannot see seq_j in K when j > i.
        # In other words, do not let earlier words (in Q) attend to later words (in K).
        if self.enable_causal_mask:
            assert in_seq_len_Q == in_seq_len_K # Should not enter this block during cross-attention
            neg_inf_matrix = torch.full_like(Q_KT_scaled[0][0], float('-inf'))
            neg_inf_mask = neg_inf_matrix.triu(diagonal = 1) # if called on 4D tensor, triu applies batch*h times to each (seq, seq) dimension, only operates on 2D matrix
            Q_KT_scaled += neg_inf_mask # if 2D tensor, (seq, seq) will broadcast into (batch_size, h, seq, seq) to mask each (seq, seq)

        # PAD MASK
        assert Q_pad_mask.ndim == K_pad_mask.ndim == 2
        Q_batch_seq_lengths = Q_pad_mask.sum(dim = 1) # length of each sequence in batch (excluding PAD_TOKEN_IDX, see top of Transformer forward for creation)
        K_batch_seq_lengths = K_pad_mask.sum(dim = 1)

        assert Q_batch_seq_lengths.size() == K_batch_seq_lengths.size() == (in_batch_size,)

        Q_seq_indices = torch.arange(in_seq_len_Q) 
        K_seq_indices = torch.arange(in_seq_len_K)

        # For each batch element, (1, seq_len) < (batch_seq_lengths, 1)
        # Each row represents a batch element. So, for each row, the first L elements will be True, and the remaining max_batch_seq_len - L will be False.
        Q_non_padding = Q_seq_indices.unsqueeze(0) < Q_batch_seq_lengths.unsqueeze(1) 
        K_non_padding = K_seq_indices.unsqueeze(0) < K_batch_seq_lengths.unsqueeze(1) 

        assert Q_non_padding.size() == (in_batch_size, in_seq_len_Q)
        assert K_non_padding.size() == (in_batch_size, in_seq_len_K)

        # For each batch element (i.e. row) in the above, we have a list of Q T/F values and a list of K T/F values.
        # Within the batch, our Q T/F values become shape (in_seq_len_Q, 1) and our K T/F values become shape (1, in_seq_len_K).
        # When we & these together (for each batch element, which is a whole sequence), we end up with a matrix of True where there is an overlap and False otherwise.
        non_padding = Q_non_padding.unsqueeze(2) & K_non_padding.unsqueeze(1) 

        assert non_padding.size() == (in_batch_size, in_seq_len_Q, in_seq_len_K)

        non_padding_h = non_padding.unsqueeze(1) # shape = (batch, 1, in_seq_len_Q, in_seq_len_K)

        assert non_padding_h.size() == (in_batch_size, 1, in_seq_len_Q, in_seq_len_K)

        Q_KT_scaled_masked = Q_KT_scaled.masked_fill(~non_padding_h, float('-inf'))
        # Q_KT_scaled_masked = Q_KT_scaled.masked_fill(~non_padding, float('-inf')) # NOTE: PyTorch should understand to broadcast (B, Q, K) pad -> (B, 1, Q, K) -> (B, H, Q, K)

        assert Q_KT_scaled_masked.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, in_seq_len_K)

        # softmax should be applied row-wise on the (t_seq, s_seq) internal matrix. Thus, batch_size * h * t_seq softmaxes will occur.
        # dim = -1 used to pull out and softmax each row (by collapsing in the column/last dimension).
        importances = torch.softmax(Q_KT_scaled_masked, dim = -1)

        assert importances.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, in_seq_len_K)

        importances = importances.nan_to_num() # Fill in bottom rows nan output from softmax with 0

        # Recall that importances is of shape (batch_size, h, seq, seq)
        # Recall that V_h_stack is of shape (batch_size, h, seq, d_v = d_k)
        # To compute each head's final attention, the last two dimensions must
        # be matrix multiplied by each other, again batch_size * h times to
        # produce a tensor of size (batch_size, h, seq, d_v).

        h_attention = importances @ V_h_stack

        assert h_attention.size() == (in_batch_size, self.num_attention_heads, in_seq_len_Q, self.d_v)

        # The final linear W_O maps between d_model dimensions. Thus, we must
        # merge the h final h_attention's horizontally back into a shape
        # of (batch_size, seq, d_model).

        H: torch.Tensor = h_attention.permute(0, 2, 1, 3).contiguous().view(in_batch_size, in_seq_len_Q, -1) # contiguous() call requires because permute breaks data continuity.

        assert H.size() == in_Q.size()
        
        MHA = self.W_O(H)

        assert MHA.size() == in_Q.size()

        return MHA