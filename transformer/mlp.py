import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Fully-connected feed-forward network to be used at the end of each
    encoder/decoder sequence.
    """
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module, dropout: float):
        """
        FeedForward network layer initializer.

        Args:
            d_model - The embedding & hidden dimension of the transformer
            d_ff - The feed-forward layer's hidden dimension 
            activation - The activation function to use in the hidden linear layer at the end of each encoder/decoder block
            dropout - The dropout percentage for the network
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(self.dropout)

        self.layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            self.activation,
            self.dropout_layer,
            nn.Linear(self.d_ff, self.d_model)
        )

    def forward(self, x: torch.Tensor):
        """
        Applies the FeedForward network to each element in the batch independently.

        Args:
            x - A single batch element, of shape (seq, d_model).

        Returns:
            o - The transformed single batch element, of shape (seq, d_model).
        """
        o: torch.Tensor = self.layers(x)

        assert x.size() == o.size()

        return o