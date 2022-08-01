import ml_train
import ml_structures
import torch

DATASET_FILE_PATH = "AL29_psth250.mat"
PARAMETER_FILE_PATH = "signal_net.pth"  # trained model parameters file

# Signals are inputs, and labels are outputs
TRAIN_SIGNAL_CACHE_FILE = "train_signals.pt"
TRAIN_LABEL_CACHE_FILE = "train_labels.pt"
TEST_SIGNAL_CACHE_FILE = "test_signals.pt"
TEST_LABEL_CACHE_FILE = "test_labels.pt"


def pre_process_data():
    """
    load the training and testing datasets from a Matlab .mat file, and save the resulting data tensors as separate files
    """
    signals, labels = ml_structures.load_data_from_mat(DATASET_FILE_PATH, 0, 855)
    train_signals = signals[0:655, :, 0:85]
    train_labels = labels[0:655]
    torch.save(train_signals, TRAIN_SIGNAL_CACHE_FILE)
    torch.save(train_labels, TRAIN_LABEL_CACHE_FILE)

    test_signals = signals[655:855, :, 0:85]
    test_labels = labels[655:855]
    torch.save(test_signals, TEST_SIGNAL_CACHE_FILE)
    torch.save(test_labels, TEST_LABEL_CACHE_FILE)


def load_saved_data(training=True):
    """
    :param bool training: whether the data loaded is training data or testing data
    """
    if training:
        signal_file = TRAIN_SIGNAL_CACHE_FILE
        label_file = TRAIN_LABEL_CACHE_FILE
    else:
        signal_file = TEST_SIGNAL_CACHE_FILE
        label_file = TEST_LABEL_CACHE_FILE
    signals = torch.load(signal_file)
    labels = torch.load(label_file)
    return signals, labels


def train_and_test_model():
    """
    train the model on the training data and then test it on the testing data
    """
    train_signals, train_labels = load_saved_data(training=True)
    test_signals, test_labels = load_saved_data(training=False)
    trainset = ml_structures.Neuron09Dataset(train_signals, train_labels)
    testset = ml_structures.Neuron09Dataset(test_signals, test_labels)
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 50
    ml_train.train_and_save(trainset, testset, PARAMETER_FILE_PATH, num_epochs, loss_fn)
    ml_train.load_and_test(testset, PARAMETER_FILE_PATH, loss_fn)


def main():
    pre_process_data()
    train_and_test_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
