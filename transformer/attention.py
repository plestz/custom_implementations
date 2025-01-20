import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism component of the transformer's encoder
    and decoder blocks.
    """
    def __init__(self, d_model: int, num_attention_heads: int):
        """
        MultiHeadAttention initializer.

        Args:
            d_model -- The embedding & hidden dimension of the transformer
            num_attention_heads -- The number of attention heads
        """
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.d_k = self.d_v = self.d_model / self.num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model) 

        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, E: torch.Tensor):
        """
        Pushes the input embedding E through a multi-head attention mechanism,
        returning a tensor of the same dimensions. The embeddings in the output
        now all contain context about the surrounding words, in addition to their
        own embedding and positional encoding.

        Args:
            E - The input embedding to be passed through the attention mechanism

        Returns:
            MHA - The result of a Multi-head Attention mechanism on E
        """
        Q, K, V = E.clone(), E.clone(), E.clone()
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)

        Q_h = list(torch.tensor_split(Q, self.num_attention_heads, dim = 1))
        K_h_T = list(map(torch.transpose, torch.tensor_split(K, self.num_attention_heads, dim = 1)))
        V_h = list(torch.tensor_split(V, self.num_attention_heads, dim = 1))

        assert len(Q_h) == len(K_h_T) == len(V_h)

        Q_h_stack = torch.stack(Q_h, dim = 0)
        K_h_T_stack = torch.stack(K_h_T, dim = 0)
        V_h_stack = torch.stack(V_h, dim = 0)

        Q_KT = torch.bmm(Q_h_stack, K_h_T_stack)
        importances = torch.softmax(Q_KT / torch.sqrt(self.d_k), dim = 1)

        h_attention = torch.bmm(importances, V_h_stack)
        H = torch.concat(torch.flatten(h_attention, end_dim = 0), dim = 1)
        MHA = self.W_O(H)

        return MHA