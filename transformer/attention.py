import torch
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    """
    """
    def __init__(self):
        """
        """

    def forward(self, x):
        """
        """
    
class MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self, num_attention_heads: int):
        """

        Args:
            num_attention_heads -- The number of attention heads used
        """
        self.num_attention_heads = num_attention_heads
        
    def forward(self, x):
        """
        """
        return x