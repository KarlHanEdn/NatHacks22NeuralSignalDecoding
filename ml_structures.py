"""
This file defines machine learning data structures needed for processing and decoding neural signals
"""

import torch
from torch.utils.data import Dataset
from torch import nn
import scipy
import scipy.io

DIM_TIME = 251  # number of evenly spaced observations in a trial; represents the time axis
NUM_LABELS = 10  # number of types of labels (10 types from 2 to 11)

# preprocessing options
APPEND_TIME_INTERVAL = 0
APPEND_INV_TIME_INTERVAL = 4
APPEND_AVG_OVER_TIME = True
APPEND_AVG_OVER_NEURONS = True
APPEND_TIME_INTERVAL_STATISTIC = (0, 8)


def my_device():
    """
    return the compatible device for the current computer
    """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data_from_mat(file_path, idx_start, idx_end):
    # load matrices from .mat file into python
    # StackOverflow answer by Gilad Naor, link:
    # https: // stackoverflow.com / questions / 874461 / read - mat - files - in -python
    mat = scipy.io.loadmat(file_path)
    signals = torch.from_numpy(mat["tc_spk"])[idx_start:idx_end]
    labels = torch.flatten(torch.from_numpy(mat["tc_stim"]))[idx_start:idx_end]
    num_trials = labels.size(dim=0)
    converted_labels = torch.zeros((num_trials, NUM_LABELS))

    # convert labels from integer to arrays of 0s with a single 1
    for i in range(num_trials):
        converted_labels[i, labels[i].item() - 2] = 1
    pass
    return signals, converted_labels


class Neuron09Dataset(Dataset):
    """
    represents a neural signal recording dataset
    responsible for keeping the inputs and outputs of the dataset as well as caching
    the entire dataset and its corresponding labels into GPU memory
    """

    def __init__(self, signals, labels):
        # to gpu device if available
        device = my_device()
        self.signals = signals.to(device)
        self.labels = labels.to(device)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

    def get_signals(self):
        return self.signals

    def get_labels(self):
        return self.labels

    def get_time_width(self):
        return self.signals.size(dim=1)

    def get_num_neurons(self):
        return self.signals.size(dim=2)


class SignalNet(nn.Module):
    POOL_PARAM = 5  # pooling parameter over time dimension

    def __init__(self, dim_time, num_neurons, is_train=True):
        super().__init__()
        self.dim_time = dim_time
        self.is_train = is_train
        self.pool = nn.MaxPool1d(SignalNet.POOL_PARAM)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_neurons * (dim_time // SignalNet.POOL_PARAM), 10)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
