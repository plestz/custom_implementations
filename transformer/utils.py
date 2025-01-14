import torch
import torch.nn.functional as F

def pad_batch_to_longest(batch: list[torch.Tensor], pad_value) -> torch.Tensor:
    """
    Given a batch of sequences (each viewed as an n_tokens amount of d_embedding vectors),
    pads each sequence such that its number of n_tokens vectors is equal to the maximum
    n_tokens length of all instances within the batch.

    Note: On input, n_tokens is unlikely to be equal among all batch elements.
    The purpose of this function is to correct this issue, so the transformer
    can process these instances in parallel.

    Args:
        batch -- A 1D list of 2D tensor of shape (d_embedding, n_tokens_i). Each list element is batch example i.
        pad_value -- The value with which to pad (on to the end of) each example.

    Returns:
        padded_batch -- A 3D tensor of shape (batch_size, d_embedding, max_n_tokens)
    """
    max_n_tokens = max(batch[i].size(dim = 1) for i in range(len(batch)))

    padded_sequences = [
        F.pad(batch[i], (0, max_n_tokens - batch[i].size(dim = 1)), mode = 'constant', value = pad_value) 
        for i in range(len(batch))
    ]

    return torch.stack(padded_sequences, dim = 0)


if __name__ == '__main__':

    sequences = [
        torch.randn(3, 3),
        torch.randn(3, 2),
        torch.randn(3, 4)
    ]

    print(sequences)

    # Verifies that output is sequences, with pad_value columns appended in the last dimension
    print(pad_batch_to_longest(sequences, pad_value = 0))

    # torch.Size([batch_size = 3, d_embedding = 3, max_n_tokens = 4])
    print(pad_batch_to_longest(sequences, pad_value = 0).shape)