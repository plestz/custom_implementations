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

def padding_collate_fn(batch: list[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]], pad_token_idx) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Given a batch of sequences (each with a source, target, and label) of varying lengths, 
    pad each set of sources, targets, and labels, up to their respective maximum length among all sequences.

    Args:
        batch - The batch of variable-length sequences to be padded

    Returns:
        (source_tensor, target_tensor) - The padded sources and targets for the input sequences
        label_tensor - The padded labels for the input sequences
    """
    num_sequences = len(batch)

    sources = [source for (source, _), _ in batch]
    targets = [target for (_, target), _ in batch]
    labels = [label for _, label in batch]

    max_source_length = max(len(tensor) for tensor in sources)
    max_target_length = max(len(tensor) for tensor in targets)
    max_label_length = max(len(tensor) for tensor in labels)

    # Pre-allocate tensors
    source_tensor = torch.full((num_sequences, max_source_length), pad_token_idx, dtype = torch.long)
    target_tensor = torch.full((num_sequences, max_target_length), pad_token_idx, dtype = torch.long)
    label_tensor = torch.full((num_sequences, max_label_length), pad_token_idx, dtype = torch.long)

    for i, source_seq in enumerate(sources):
        source_tensor[i, :len(source_seq)] = source_seq

    for i, target_seq in enumerate(targets):
        target_tensor[i, :len(target_seq)] = target_seq

    for i, label_seq in enumerate(labels):
        label_tensor[i, :len(label_seq)] = label_seq

    return (source_tensor, target_tensor), label_tensor

def list_padding_collate_fn(batch: list[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]], pad_token_idx) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Given a batch of sequences (each with a source, target, and label) of varying lengths, 
    pad each set of sources, targets, and labels, up to their respective maximum length among all sequences.

    Args:
        batch - The batch of variable-length sequences to be padded

    Returns:
        (source_tensor, target_tensor) - The padded sources and targets for the input sequences
        label_tensor - The padded labels for the input sequences
    """
    num_sequences = len(batch)

    sources = [torch.tensor(source) for (source, _), _ in batch]
    targets = [torch.tensor(target) for (_, target), _ in batch]
    labels = [torch.tensor(label) for _, label in batch]

    max_source_length = max(len(tensor) for tensor in sources)
    max_target_length = max(len(tensor) for tensor in targets)
    max_label_length = max(len(tensor) for tensor in labels)

    # Pre-allocate tensors
    source_tensor = torch.full((num_sequences, max_source_length), pad_token_idx, dtype = torch.long)
    target_tensor = torch.full((num_sequences, max_target_length), pad_token_idx, dtype = torch.long)
    label_tensor = torch.full((num_sequences, max_label_length), pad_token_idx, dtype = torch.long)

    for i, source_seq in enumerate(sources):
        source_tensor[i, :len(source_seq)] = source_seq

    for i, target_seq in enumerate(targets):
        target_tensor[i, :len(target_seq)] = target_seq

    for i, label_seq in enumerate(labels):
        label_tensor[i, :len(label_seq)] = label_seq

    return (source_tensor, target_tensor), label_tensor