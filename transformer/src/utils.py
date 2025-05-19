import torch
import torch.nn.functional as F

import numpy as np

from typing import Union

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

def padding_collate_fn(batch: list[tuple[tuple[Union[torch.Tensor, list], Union[torch.Tensor, list]], Union[torch.Tensor, list]]], pad_token_idx) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Given a batch of sequences (each with a source, target, and label) of varying lengths, 
    pad each set of sources, targets, and labels, up to their respective maximum length among all sequences.

    Each source, target, and label must be of type torch.Tensor or list.

    Args:
        batch - The batch of variable-length sequences to be padded

    Returns:
        (source_tensor, target_tensor) - The padded sources and targets for the input sequences
        label_tensor - The padded labels for the input sequences
    """
    num_sequences = len(batch)

    # NOTE: Assumes, within any batch element, label shares type with source and target.    
    if isinstance(batch[0][1], torch.Tensor):
        cast_func = lambda x: x # identity function
    elif isinstance(batch[0][1], list):
        cast_func = lambda x: torch.tensor(x)
    else:
        raise TypeError('Batch element does not contain source, target, and label of supported type.')

    sources = [cast_func(source) for (source, _), _ in batch]
    targets = [cast_func(target) for (_, target), _ in batch]
    labels = [cast_func(label) for _, label in batch]

    max_source_length = max(len(tensor) for tensor in sources)
    max_target_length = max(len(tensor) for tensor in targets)
    max_label_length = max(len(tensor) for tensor in labels)

    # NOTE: Future improvement: use torch.nn.utils.rnn.pad_sequence for vectorized operation
    # Pre-allocate tensors
    source_tensor = torch.full((num_sequences, max_source_length), pad_token_idx, dtype = torch.long)
    target_tensor = torch.full((num_sequences, max_target_length), pad_token_idx, dtype = torch.long)
    label_tensor = torch.full((num_sequences, max_label_length), pad_token_idx, dtype = torch.long)

    for i, (source_seq, target_seq, label_seq) in enumerate(zip(sources, targets, labels)):
        source_tensor[i, :len(source_seq)] = source_seq
        target_tensor[i, :len(target_seq)] = target_seq
        label_tensor[i, :len(label_seq)] = label_seq

        if len(target_seq) != len(label_seq):
            print(source_seq)
            print(target_seq)
            print(label_seq)
            raise ValueError('force break') 

    return (source_tensor, target_tensor), label_tensor