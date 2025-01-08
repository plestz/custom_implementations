import torch
import torch.nn as nn

class UnivariateFunction(nn.Module):
    """
    A custom UnivariateFunction to resemble the per-edge
    activation function in Kolmogorov-Arnold Networks.

    With respect to the official KAN paper (linked in repository README),
    UnivariateFunction is a viable replacement for the BSpline.
    """
    def __init__(self, hidden_size = 5):
        """
        UnivariateFunction initializer. 

        Note: Given the quantity of these, be wary of greater layer sizes
        for training time.

        Args:
            hidden_size - The number of neurons in each hidden layer.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        Transforms real-valued input into real-valued output.

        Note: This accepts an optional batch dimension, e.g. (n, 1).

        Args:
            x - Real-valued activation function input.

        Returns:
            x - Real-valued activation function output.
        """
        x = self.layers(x)

        return x
    
class KANLayer(nn.Module):
    """
    Represents one full layer in a Kolmogorov-Arnold Network, which can
    take any number of input neurons and transform their values into
    any number of output neurons (with an activation function on each
    distinct edge). 
    
    Thus, the number of activation functions present
    within this layer will be equal to the number of input neurons times
    the number of output neurons.
    """
    def __init__(self, input_size, output_size, hidden_size = 5):
        """
        KANLayer Initializer. Requires input_size and output_size to ensure
        KANLayer is aware of quantity of activation functions required for
        forward pass.

        Args:
            input_size - The number of input dimensions of the instances
            to be received by this layer. This amount will by creation
            match the number of input functions feeding into each
            output node.

            output_size - The number of output dimensions to be produced
            by this layer.

            hidden_size - The number of neurons in each hidden layer of the
            activation functions.
        """
        super().__init__()

        # OUTPUT rows by INPUT columns matrix of functions
        self.activation_matrix = nn.ModuleList(
        [nn.ModuleList([UnivariateFunction(hidden_size) for _ in range(input_size)])]
        for _ in range(output_size))

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Performs a full one-layer forward pass through the KAN Network, where
        each distinct edge is represented by an activation function.

        More specifically, for any particular example i, for each output node o_k, 
        this systematically composes each input x_{ij} with f_j such that
        o_k = \sum{j=1}^{\# input features} f_j(x_ij). Thus, each instance within
        a batch will have a corresponding k output nodes.

        Args:
            x - Input, passed as 2D tensor: Batch x Number of Features 

        Returns:
            x - Output, returned as 2D tensor: Batch x Number of Output Nodes (output_size)
        """
        # Assert that the number of input dimensions equals the number of input functions
        assert len(x[0]) == self.input_size

        batch_size = len(x)

        row_batch_activations = list()

        for output_node_idx in range(self.output_size):

            total = torch.zeros(batch_size, 1)

            for input_node_idx in range(self.input_size):

                feature_col = x[:, input_node_idx].unsqueeze(1)
                total += self.activation_matrix[output_node_idx][input_node_idx](feature_col)

            row_batch_activations.append(total)

        activations = torch.cat(row_batch_activations, dim = 1)

        return activations