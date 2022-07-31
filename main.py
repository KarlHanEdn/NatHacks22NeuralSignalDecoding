import ml_train
import ml_structures
import torch

DATASET_FILE_PATH = "AL29_psth250.mat"
PROCESSED_SIGNAL_CACHE_FILE = "processed_signals.pt"
LABEL_CACHE_FILE = "processed_labels.pt"
PARAMETER_FILE_PATH = "signal_net.pth"


def pre_process_data():
    signals, labels = ml_structures.load_data_from_mat(DATASET_FILE_PATH, 0, 855)
    processed_signals = ml_structures.Neuron09Dataset.pre_process_signals(signals)
    print(processed_signals[0, :, 1])
    torch.save(processed_signals, PROCESSED_SIGNAL_CACHE_FILE)
    torch.save(labels, LABEL_CACHE_FILE)


def load_pre_process_data():
    signals = torch.load(PROCESSED_SIGNAL_CACHE_FILE)
    labels = torch.load(LABEL_CACHE_FILE)
    return signals, labels


def train_and_test_model_from_dataset(signals, labels):
    trainset = ml_structures.Neuron09Dataset(signals[0:655], labels[0:655])
    testset = ml_structures.Neuron09Dataset(signals[655:855], labels[655:855])
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 50
    ml_train.train_and_save(trainset, testset, PARAMETER_FILE_PATH, num_epochs, loss_fn)
    ml_train.load_and_test(testset, PARAMETER_FILE_PATH, loss_fn)


def main():
    # pre_process_data()
    signals, labels = load_pre_process_data()
    train_and_test_model_from_dataset(signals[:, 0:252, :], labels)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
