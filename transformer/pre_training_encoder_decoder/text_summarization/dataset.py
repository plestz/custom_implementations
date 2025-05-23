import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TextSummarizationDataset(Dataset):
    """
    A dataset the houses the sources, targets, and labels for an 
    Encoder-Decoder Transformer summarization task.
    """
    def __init__(self, sources: list, targets: list, labels: list):
        """
        Initializes the TextSummarizationDataset.
        """
        self.sources = sources
        self.targets = targets
        self.labels = labels

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.sources)

    def __getitem__(self, idx):
        """
        Returns the source, target, and label sequences at the specified index.
        """
        # print((self.sources[idx], self.targets[idx]), self.labels[idx])
        return (self.sources[idx], self.targets[idx]), self.labels[idx]