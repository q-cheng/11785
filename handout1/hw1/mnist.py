"""Problem 3 - Training on MNIST"""
import numpy as np

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
from mytorch.nn.batchnorm import BatchNorm1d
from mytorch.nn.activations import ReLU
from mytorch.nn.linear import Linear
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.nn.sequential import Sequential
from mytorch.optim.sgd import SGD
from mytorch.tensor import Tensor

BATCH_SIZE = 100


def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)

    Args:
        train_x (np.array): training data (55000, 784)
        train_y (np.array): training labels (55000,)
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    my_model = Sequential(Linear(784, 20), BatchNorm1d(20), ReLU(), Linear(20, 10))
    # TODO: Call training routine (make sure to write it below)
    optimizer = SGD(my_model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()
    val_accuracies = train(my_model, optimizer, criterion, train_x, train_y, val_x, val_y)
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []

    # TODO: Implement me! (Pseudocode on writeup)
    for epoch in range(num_epochs):
        model.train()
        np.random.seed(11785)
        np.random.shuffle(train_x)
        np.random.seed(11785)
        np.random.shuffle(train_y)
        batches_x = np.split(train_x, train_x.shape[0] // BATCH_SIZE)
        batches_y = np.split(train_y, train_y.shape[0] // BATCH_SIZE)
        for i, (batch_data, batch_label) in enumerate(zip(batches_x, batches_y)):
            optimizer.zero_grad()
            tensor_data = Tensor(batch_data)
            tensor_label = Tensor(batch_label)
            out = model(tensor_data)
            loss = criterion(out, tensor_label)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
    return val_accuracies


def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    # TODO: implement validation based on pseudocode
    model.eval()
    batches_x = np.split(val_x, val_x.shape[0] // BATCH_SIZE)
    batches_y = np.split(val_y, val_y.shape[0] // BATCH_SIZE)
    # accuracy = 0
    cnt = 0
    for datas, labels in zip(batches_x, batches_y):
        tensor_x = Tensor(datas)
        out = model(tensor_x)
        batch_preds = np.argmax(out.data, axis=1)
        # print(out.data)
        for res, res_ref in zip(batch_preds, labels):
            if res == res_ref:
                cnt += 1
        # print(cnt)
        # accuracy += cnt / len(labels)
    # return accuracy / (val_y.shape[0] // BATCH_SIZE)
    model.train()
    return cnt / val_y.shape[0]
