import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TextSummarizationDataset(Dataset):
    """
    """
    def __init__(self, sources: list, targets: list, labels: list):
        self.sources = sources
        self.targets = targets
        self.labels = labels

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        print((self.sources[idx], self.targets[idx]), self.labels[idx])
        return (self.sources[idx], self.targets[idx]), self.labels[idx]