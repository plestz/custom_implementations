import torch
from torch.utils.data import Dataset
import random

class RandomSortedIntegerDataset(Dataset):
    """
    A dataset that generates random sorted integer sequences.
    """
    def __init__(self, min_seq_len: int, num_sequences: int, vocab: list):
        """
        Initializes the RandomSortedIntegerDataset.
        """
        self.inputs, self.labels = generate_random_sorted_integer_sequences(min_seq_len, num_sequences, vocab)

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns the input and label sequences at the specified index.
        """
        return self.inputs[idx], self.labels[idx]

def generate_random_sorted_integer_sequences(min_seq_len: int, num_sequences: int, vocab: list) -> tuple[tuple[list[torch.Tensor], list[torch.Tensor]], list[torch.Tensor]]:
    """
    Generate sorted integer sequences of a size between min_seq_len and vocab_size 
    (both inclusive), where elements are from vocab.

    This function does not perform any sequence padding.

    Args:
        min_seq_len - The minimum sequence length that can be generated
        num_sequences - The number of sequences to be generated
        vocab - The vocabulary from which to pull sequence elements.

    Returns:
        inputs - The list of inputs (for the encoder)
        labels - The list of labels (for comparison with the encoder MLM output)
    """
    inputs = list() # encoder inputs
    labels = list() # ground truths

    vocab_size = len(vocab) # indices [0, ... , len(vocab) - 1]
    PAD_TOKEN_IDX = vocab_size
    CLS_TOKEN_IDX = vocab_size + 1
    SEP_TOKEN_IDX = vocab_size + 2
    MASK_TOKEN_IDX = vocab_size + 3

    for _ in range(num_sequences):
        seq_len = random.randint(min_seq_len, vocab_size) # inclusive of start and end
        random_shift = random.randint(0, vocab_size - seq_len)
        sorted_seq = [i + random_shift for i in range(0, seq_len)]

        input = torch.tensor([CLS_TOKEN_IDX] + sorted_seq + [SEP_TOKEN_IDX])
        label = torch.tensor([MASK_TOKEN_IDX] + sorted_seq + [MASK_TOKEN_IDX])

        inputs.append(input)
        labels.append(label)

    return inputs, labels
