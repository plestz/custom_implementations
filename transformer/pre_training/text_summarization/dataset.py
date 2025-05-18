import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TextSummarizationDataset(Dataset):
    """
    """
    def __init__(self, sequences: pd.Series, summaries: pd.Series):
        self.sequences = sequences
        self.summaries = summaries

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.summaries[idx]