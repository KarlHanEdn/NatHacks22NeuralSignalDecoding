"""
This file defines machine learning data structures needed for processing and decoding neural signals
"""

import torch
from torch.utils.data import Dataset
from torch import nn
import scipy
import scipy.io
import math
import random

NUM_NEURONS = 85
DIM_TIME = 251  # number of evenly spaced observations in a trial; represents the time axis
NUM_LABELS = 10  # number of types of labels (10 types from 2 to 11)

# preprocessing options
APPEND_TIME_INTERVAL = 0
APPEND_INV_TIME_INTERVAL = 4
APPEND_AVG_OVER_TIME = True
APPEND_AVG_OVER_NEURONS = True
APPEND_TIME_INTERVAL_STATISTIC = (0, 8)


def compute_preprocessed_data_height():
    height = DIM_TIME + APPEND_TIME_INTERVAL + APPEND_INV_TIME_INTERVAL
    if APPEND_AVG_OVER_TIME:
        height += 1
    if APPEND_AVG_OVER_NEURONS:
        height += math.ceil(DIM_TIME / NUM_NEURONS)  # reshape the row into a column and then append it
    stat = APPEND_TIME_INTERVAL_STATISTIC
    height += max(0, stat[1] - stat[0])

    return height


DIM_TIME_PROCESSED = compute_preprocessed_data_height()  # after data processing


def my_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data_from_mat(file_path, idx_start, idx_end):
    # load matrices from .mat file into python
    # StackOverflow answer by Gilad Naor, link:
    # https: // stackoverflow.com / questions / 874461 / read - mat - files - in -python
    mat = scipy.io.loadmat(file_path)
    signals = torch.from_numpy(mat["tc_spk"])[idx_start:idx_end]
    labels = torch.flatten(torch.from_numpy(mat["tc_stim"]))[idx_start:idx_end]
    return signals, labels


class Neuron09Dataset(Dataset):
    """
    represents a neural signal recording dataset
    responsible for reading the dataset and labels from a .mat file, pre-processing them as well as caching
    the entire dataset and its corresponding labels into GPU memory
    also responsible for performing data augmentation in the pre-processing step
    """

    def __init__(self, signals, labels):
        # to gpu device if available
        self.signals = signals
        self.labels = labels

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

    def get_signals(self):
        return self.signals

    def get_labels(self):
        return self.labels

    def get_time_width(self):
        return self.signals.size(dim=1)

    @staticmethod
    def pre_process_signals(signals):
        print("Pre-processing...")
        num_trials = signals.size(dim=0)
        processed_signals = torch.zeros((num_trials, DIM_TIME_PROCESSED, NUM_NEURONS))
        processed_signals[:, 0:DIM_TIME, :] = signals[:, 0:DIM_TIME, :]
        print(f"Copied raw data: idx {0} to {DIM_TIME - 1} ")

        # perform aggregation to obtain stats about the data
        avg_scale_factor = 10
        cur_idx = DIM_TIME  # where we start appending data
        if APPEND_AVG_OVER_TIME:
            print(f"Avg over time: idx {cur_idx}")
            avg_over_time = torch.mean(signals[:, 0:DIM_TIME, :], dim=1)
            processed_signals[:, cur_idx, :] = avg_over_time * avg_scale_factor
            cur_idx += 1
        if APPEND_AVG_OVER_NEURONS:
            width = math.ceil(DIM_TIME / NUM_NEURONS)
            print(f"Avg over neurons: idx {cur_idx} to {cur_idx + width - 1} ")
            padding = width * NUM_NEURONS - DIM_TIME
            avg_over_neurons = torch.mean(signals[:, 0:DIM_TIME, :], dim=2) * avg_scale_factor
            avg_over_neurons = torch.cat((avg_over_neurons, torch.zeros((num_trials, padding))), dim=1)
            processed_signals[:, cur_idx:cur_idx + width, :] = \
                avg_over_neurons.reshape((num_trials, width, NUM_NEURONS))
            cur_idx += width

        stats = APPEND_TIME_INTERVAL_STATISTIC
        stats_len = stats[1] - stats[0]
        total_append = APPEND_TIME_INTERVAL + APPEND_INV_TIME_INTERVAL + stats_len
        for i in range(num_trials):
            if i % 100 == 99:
                print(f"trial {i}")
            for j in range(NUM_NEURONS):
                inactive_obs = 0  # number of consecutive inactive signals of this particular neuron
                total_active_obs = 0
                time_intervals = torch.zeros((APPEND_TIME_INTERVAL, ))
                inv_time_intervals = torch.zeros((APPEND_INV_TIME_INTERVAL, ))
                time_interval_stats = torch.zeros((stats_len, ))
                for k in range(DIM_TIME):
                    if signals[i, k, j].item() < 0.5:
                        inactive_obs += 1
                    else:
                        if total_active_obs < APPEND_TIME_INTERVAL:
                            time_intervals[total_active_obs] = inactive_obs * 0.04
                        if total_active_obs < APPEND_INV_TIME_INTERVAL:
                            inv_time_intervals[total_active_obs] = 10 / (inactive_obs + 5)
                        course_count = int(inactive_obs / 6)
                        if stats[0] <= course_count < stats[1]:
                            time_interval_stats[course_count - stats[0]] += 1
                        total_active_obs += 1
                        inactive_obs = 0
                processed_signals[i, cur_idx:cur_idx + total_append, j] = \
                    torch.cat((time_intervals, inv_time_intervals, time_interval_stats))

        print(f"Time intervals: idx {cur_idx} to {cur_idx + APPEND_TIME_INTERVAL - 1} ")
        cur_idx += APPEND_TIME_INTERVAL
        print(f"Inv intervals: idx {cur_idx} to {cur_idx + APPEND_INV_TIME_INTERVAL - 1} ")
        cur_idx += APPEND_INV_TIME_INTERVAL
        print(f"Stats of interval lengths: idx {cur_idx} to {cur_idx + stats_len - 1} ")
        cur_idx += stats_len

        assert cur_idx == DIM_TIME_PROCESSED

        return processed_signals


class SignalNet(nn.Module):
    POOL_PARAM = 5

    def __init__(self, dim_time):
        super().__init__()
        self.dim_time = dim_time
        self.pool = nn.MaxPool1d(SignalNet.POOL_PARAM)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(NUM_NEURONS * dim_time, 10)

    def forward(self, x):
        # if x.size(dim=1) > DIM_TIME:
        #     x = x.transpose(1, 2)
        #     y = self.pool(x[:, :, 0:DIM_TIME])
        #     x = torch.cat((y, x[:, :, DIM_TIME:-1]))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
