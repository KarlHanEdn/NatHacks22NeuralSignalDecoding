"""
This file defines the training loop and testing loop for the network
see also pytorch official tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""
import torch
from ml_structures import NUM_LABELS
from sklearn import svm
from joblib import dump, load

BATCH_SIZE = 4  # training mini-batch size


def create_model():
    model = svm.SVC(decision_function_shape='ovo', kernel='linear')
    return model


def train_and_save(trainset, testset, file_path):
    model = create_model()

    signals = trainset.get_signals()
    labels = trainset.get_labels()
    signals, labels = data_torch_to_numpy(signals, labels)
    model.fit(signals, labels)
    print("Finished training\n")

    # save the model parameters
    dump(model, file_path)


def load_and_test(testset, file_path):
    model = load(file_path)
    detailed_test(testset, model)


def data_torch_to_numpy(signals, labels):
    return torch.nn.Flatten()(signals).numpy(), labels.numpy()


def test_loop(testset, model):
    """
    test the loss and accuracy of the model on the entire testset
    :param Neuron09Dataset testset:
    :param nn.Module model:
    """
    signals = testset.get_signals()
    labels = testset.get_labels()
    testset_size = signals.size(dim=0)
    signals, labels = data_torch_to_numpy(signals, labels)

    num_correct = 0
    predicted = model.predict(signals)
    for i in range(testset_size):
        if predicted[i] == labels[i]:
            num_correct += 1
    print(f"test accuracy: {num_correct / testset_size * 100}%")


def detailed_test(testset, model):
    """
    a more detailed version of test_loop(), used in evaluating model after training
    :param Neuron09Dataset testset:
    :param nn.Module model:
    """
    signals = testset.get_signals()
    labels = testset.get_labels()
    testset_size = signals.size(dim=0)
    signals, labels = data_torch_to_numpy(signals, labels)

    # display a sample of model outputs on a set of input
    sample_size = 10
    predicted_samples = model.predict(signals[0:sample_size]).tolist()
    predicted_samples = [str(item) for item in predicted_samples]
    actual_samples = labels[0:sample_size].tolist()
    actual_samples = [str(item) for item in actual_samples]
    print(f"predicted samples: {' '.join(predicted_samples)}")
    print(f"actual samples: {' '.join(actual_samples)}")

    # compute accuracy of each type of label on the test set
    correct_pred = {i: 0 for i in range(2 + NUM_LABELS)}
    total_pred = {i: 0 for i in range(2 + NUM_LABELS)}
    predicted = model.predict(signals)
    for i in range(testset_size):
        actual = labels[i]
        if predicted[i] == actual:
            correct_pred[actual] += 1
        total_pred[actual] += 1
    for i in range(NUM_LABELS):
        if total_pred[i] != 0:
            print(f"Accuracy for label type {i}: {correct_pred[i] / total_pred[i] * 100}%")

    test_loop(testset, model)
