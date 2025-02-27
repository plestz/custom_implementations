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
        print(self.inputs[0].shape)
        print(self.inputs[1].shape)
        print(self.labels.shape)

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns the input and label sequences at the specified index.
        """
        return (self.inputs[0][idx], self.inputs[1][idx]), self.labels[idx]

def generate_random_integer_sequences(min_seq_len: int, max_seq_len: int, num_sequences: int, vocab: list):
    """
    """
    sources = list() # encoder inputs
    targets = list() # decoder inputs
    labels = list() # ground truths

    vocab_size = len(vocab) # indices [0,...,len(vocab) - 1]
    PAD_TOKEN_IDX = vocab_size
    START_TOKEN_IDX = vocab_size + 1
    END_TOKEN_IDX = vocab_size + 2

    for _ in range(num_sequences):
        seq_len = random.randint(min_seq_len, max_seq_len)
        seq = random.sample(vocab, seq_len)

        padding_required = max_seq_len - seq_len

        source = torch.tensor(seq + [PAD_TOKEN_IDX] * padding_required) # Sequence Length = max_seq_len
        target = torch.tensor([START_TOKEN_IDX] + sorted(seq) + [PAD_TOKEN_IDX] * padding_required) # Sequence Length = max_seq_len + 1
        label = torch.tensor(sorted(seq) + [END_TOKEN_IDX] + [PAD_TOKEN_IDX] * padding_required) # Sequence Length = max_seq_len + 1

        sources.append(source)
        targets.append(target)
        labels.append(label)

    return (torch.stack(sources), torch.stack(targets)), torch.stack(labels)
