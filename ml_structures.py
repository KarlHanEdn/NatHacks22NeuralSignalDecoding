"""
This file defines machine learning data structures needed for processing and decoding neural signals
"""

import torch
from torch.utils.data import Dataset
from torch import nn
import scipy.io

NUM_NEURONS = 85
DIM_TIME = 251  # number of evenly spaced observations in a trial; represents the time axis


class Neuron09Dataset(Dataset):
    """
    represents a neural signal recording dataset
    responsible for reading the dataset and labels from a .mat file, pre-processing them as well as caching
    the entire dataset and its corresponding labels into GPU memory
    """

    def __init__(self, file_path):
        """
        :param str file_path: path to the .mat file containing the recorded neural signal data
        """
        pass

    def __getitem__(self, idx):
        pass


class SignalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_NEURONS * DIM_TIME, 10),
        )

    def forward(self, x):
        logits = self.nn_stack(x)
        return logits
