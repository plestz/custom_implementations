import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

class RandomIntegerDataset(Dataset):
    """
    A dataset that generates random integer sequences.
    """
    def __init__(self, min_seq_len: int, max_seq_len: int, num_sequences: int, vocab: list):
        """
        Initializes the RandomIntegerDataset.
        """
        self.inputs, self.labels = generate_random_integer_sequences(min_seq_len, max_seq_len, num_sequences, vocab)

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns the source, target, and label sequences at the specified index.
        """
        return (self.inputs[0][idx], self.inputs[1][idx]), self.labels[idx]

def generate_random_integer_sequences(min_seq_len: int, max_seq_len: int, num_sequences: int, vocab: list) -> tuple[tuple[list[torch.Tensor], list[torch.Tensor]], list[torch.Tensor]]:
    """
    Generate num_sequences of a size between min_seq_len and max_seq_len (both inclusive),
    where elements are from vocab.

    This function does not perform any sequence padding.

    Args:
        min_seq_len - The minimum sequence length that can be generated
        max_seq_len - The maximum sequence length that can be generated
        num_sequences - The number of sequences to be generated
        vocab - The vocabulary from which to pull sequence elements (with replacement)

    Returns:
        (source_tensor, target_tensor) - The list of sources (for the encoder input) and targets (for the decoder input).
        label_tensor - The list of labels (for comparison with the decoder output)
    """
    sources = list() # encoder inputs
    targets = list() # decoder inputs
    labels = list() # ground truths

    vocab_size = len(vocab) # indices [0,...,len(vocab) - 1]
    PAD_TOKEN_IDX = vocab_size
    SOS_TOKEN_IDX = vocab_size + 1
    EOS_TOKEN_IDX = vocab_size + 2

    for _ in range(num_sequences):
        seq_len = random.randint(min_seq_len, max_seq_len) # inclusive of start and end
        seq = random.choices(vocab, k = seq_len) # samples with replacement
        sorted_seq = sorted(seq)

        source = torch.tensor(seq)
        target = torch.tensor([SOS_TOKEN_IDX] + sorted_seq)
        label = torch.tensor(sorted_seq + [EOS_TOKEN_IDX])

        sources.append(source)
        targets.append(target)
        labels.append(label)

    return (sources, targets), labels
