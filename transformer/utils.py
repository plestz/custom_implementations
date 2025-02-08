import torch
import torch.nn.functional as F

import numpy as np

def pad_batch_to_longest_seq_len(batch: list[torch.Tensor], pad_value = 0) -> tuple[torch.Tensor, int]:
    """
    Given a batch of sequences (each viewed as a seq amount of d_embedding vectors),
    pads each sequence such that its number of seq vectors is equal to the maximum
    seq length of all instances within the batch.

    Note: On input, seq is unlikely to be equal among all batch elements.
    The purpose of this function is to correct this issue, so the transformer
    can process these instances in parallel.

    Args:
        batch -- A 1D list of 2D tensor of shape (seq, d_model). Each list element is batch example i.
        pad_value -- The value with which to pad (on to the end of) each example.

    Returns:
        padded_batch -- A 3D tensor of shape (batch_size, seq, d_model)
        max_seq - The maximum sequence length within the batch
    """
    max_seq = max(batch[i].size(dim = 0) for i in range(len(batch)))

    padded_sequences = [
        F.pad(batch[i], (0, 0, 0, max_seq - batch[i].size(dim = 0)), value = pad_value) 
        for i in range(len(batch))
    ]

    return torch.stack(padded_sequences, dim = 0).float(), max_seq


if __name__ == '__main__':

    sequences = [
        torch.randn(3, 3),
        torch.randn(2, 3),
        torch.randn(4, 3)
    ]

    print(sequences)

    # Verifies that output is sequences, with pad_value rows appended
    print(pad_batch_to_longest(sequences, pad_value = 0))

    # torch.Size([batch_size = 3, seq = 4, d_model = 3])
    print(pad_batch_to_longest(sequences, pad_value = 0).shape)