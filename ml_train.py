"""
This file defines the training loop and testing loop for the network
see also pytorch official tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""
import torch
from ml_structures import NUM_LABELS
from ml_structures import SignalNet
from ml_structures import my_device

BATCH_SIZE = 4  # training mini-batch size


def create_model():
    net = SignalNet()
    net.to(my_device())
    return net


def train_and_save(trainset, testset, file_path, num_epochs):
    model = create_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(num_epochs):
        train_loop(trainset, model, loss_fn, torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9))
        test_loop(testset, model, loss_fn)
    print("Finished training")

    # save the model parameters
    torch.save(model.state_dict(), file_path)


def train_loop(trainset, model, loss_fn, optimizer):
    """
    train the model on entire trainset for one epoch
    :param Neuron09Dataset trainset:
    :param nn.Module model:
    :param loss_fn:
    :param optimizer:
    """
    signals = trainset.getSignals()
    labels = trainset.getLabels()
    num_batches = int(signals.size(dim=0) / BATCH_SIZE)

    accumulated_loss = 0
    for i in range(num_batches):
        # take a slice of the dataset as minibatch
        batch_start = i * BATCH_SIZE
        batch_end = (i + 1) * BATCH_SIZE  # one past the end
        cur_signals = signals[batch_start:batch_end]
        cur_labels = labels[batch_start:batch_end]

        # zero and then recompute the gradient
        optimizer.zero_grad()
        outputs = model(cur_signals)
        loss = loss_fn(outputs, cur_labels)
        loss.backward()
        optimizer.step()

        accumulated_loss += loss.item()

    # print loss on training set
    print(f"train avg loss: {accumulated_loss / (num_batches * BATCH_SIZE)}")


def load_and_test(testset, file_path):
    model = create_model()
    loss_fn = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(file_path))
    detailed_test(testset, model, loss_fn)


def test_loop(testset, model, loss_fn):
    """
    test the loss and accuracy of the model on the entire testset
    :param Neuron09Dataset testset:
    :param nn.Module model:
    :param loss_fn:
    """
    signals = testset.getSignals()
    labels = testset.getLabels()
    testset_size = signals.size(dim=0)

    accumulated_loss = 0
    num_correct = 0
    with torch.no_grad():
        for i in range(testset_size):
            cur_signals = signals[i:i + 1]
            cur_labels = labels[i:i + 1]
            outputs = model(cur_signals)
            loss = loss_fn(outputs, cur_labels)
            accumulated_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.item()
            _, actual = torch.max(cur_labels, 1)
            actual = actual.item()
            if predicted == actual:
                num_correct += 1
    print(f"test avg loss: {accumulated_loss / testset_size}")
    print(f"test accuracy: {num_correct / testset_size}%")


def detailed_test(testset, model, loss_fn):
    """
    a more detailed version of test_loop(), used in evaluating model after training
    :param Neuron09Dataset testset:
    :param nn.Module model:
    :param loss_fn:
    """
    signals = testset.getSignals()
    labels = testset.getLabels()
    testset_size = signals.size(dim=0)

    # display a sample of model outputs on a set of input
    sample_size = 10
    predicted_samples = []
    actual_samples = []
    with torch.no_grad():
        for i in range(sample_size):
            cur_signals = signals[i:i + 1]
            cur_labels = labels[i:i + 1]
            outputs = model(cur_signals)

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.item()
            _, actual = torch.max(cur_labels, 1)
            actual = actual.item()
            predicted_samples.append(predicted)
            actual_samples.append(actual)

    # compute accuracy of each type of label on the test set
    correct_pred = {i: 0 for i in range(NUM_LABELS)}
    total_pred = {i: 0 for i in range(NUM_LABELS)}
    with torch.no_grad():
        for i in range(testset_size):
            cur_signals = signals[i:i + 1]
            cur_labels = labels[i:i + 1]
            outputs = model(cur_signals)

            _, predicted = torch.max(outputs, 1)
            predicted = predicted.item()
            _, actual = torch.max(cur_labels, 1)
            actual = actual.item()
            if predicted == actual:
                correct_pred[actual] += 1
            total_pred[actual] += 1

    test_loop(testset, model, loss_fn)
