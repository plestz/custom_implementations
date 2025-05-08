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
    max_seq = max(tensor.size(dim = 0) for tensor in batch)

    # Pre-allocate 3D tensor
    padded_batch = torch.full(
        (len(batch), max_seq, batch[0].size(dim = 1)), 
        pad_value, 
        dtype = batch[0].dtype
    )

    # Copy over batch sequences into pre-allocated tensor
    for i, sequence in enumerate(batch):
        padded_batch[i, :sequence.size(dim = 0)] = sequence

    return padded_batch.float(), max_seq


if __name__ == '__main__':

    sequences = [
        torch.randn(3, 3),
        torch.randn(2, 3),
        torch.randn(4, 3)
    ]

    print(sequences)

    # Verifies that output is sequences, with pad_value rows appended
    padded = pad_batch_to_longest_seq_len(sequences, pad_value = 0)

    print(padded)

    # torch.Size([batch_size = 3, seq = 4, d_model = 3])
    assert padded[0].size() == (3, 4, 3)
    assert padded[1] == 4