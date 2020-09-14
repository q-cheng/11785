from abc import ABC
from typing import Any

import numpy as np
import torch
import csv

# Constants
CONTEXT = 8
EPOCH = 10
BATCH_SIZE = 1024
NUM_WORKERS = 0
LEARNING_RATE = 0.1
OUTPUT_PATH = 'res.csv'


# MLP model
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(((1 + 2 * CONTEXT) * 13), 1024)  # Hidden layer.
        self.layer2 = torch.nn.Linear(1024, 346)  # Output layer.
        self.reLu = torch.nn.ReLU()
        
    def forward(self, x):
        x = x.view((-1, (1 + 2 * CONTEXT) * 13))
        x = self.layer1(x)
        x = self.reLu(x)
        x = self.layer2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x
    
    
# Helper function for padding matrix.
def pad_central(x):
    if len(x.shape) == 2:
        dim1 = x.shape[0]
        dim2 = x.shape[1]
        result = np.zeros((dim1 + 2 * CONTEXT, dim2))
        result[CONTEXT:-CONTEXT] = x
        
    if len(x.shape) == 1:
        dim1 = x.shape[0]
        result = np.zeros((dim1 + 2 * CONTEXT,))
        result[CONTEXT:-CONTEXT] = x
    return result


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_data, transform=None):
        self.input_data = test_data
        self.input_after_pad = pad_central(self.input_data)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        new_idx = idx + CONTEXT
        return_matrix = np.array(self.input_after_pad[new_idx - CONTEXT:new_idx + CONTEXT + 1])
        return_tensor = torch.tensor(return_matrix)
        return return_tensor


# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_npy_file, label_npy_file, transform=None):
        self.input_data = np.load(data_npy_file, allow_pickle=True)
        self.input_label = np.load(label_npy_file, allow_pickle=True)
        self.training_data = []
        # Padding with CONTEXT number of np.zeros([Y x Z]) and Flattern the frames.
        for group_idx in range(self.input_data.shape[0]):
            cur_group, cur_label = self.input_data[group_idx], self.input_label[group_idx]
            group_after_pad = pad_central(cur_group)
            label_after_pad = pad_central(cur_label)
            for pair in zip(group_after_pad, label_after_pad):
                # pair -> (1*13, label of middle sample)
                self.training_data.append(pair)

    def __len__(self):
        # print('length:', len(self.training_data) - 2 * CONTEXT)
        return len(self.training_data) - 2 * CONTEXT
    
    def __getitem__(self, idx):
        new_idx = idx + CONTEXT  # Start idx in training data is CONTEXT.
        # Train and validate.
        return_matrix = np.ones([1 + 2 * CONTEXT, 13])
        for i in range(new_idx-CONTEXT, new_idx+CONTEXT+1):
            # Transform to (1 + 2 * CONTEXT, 13)
            return_matrix[i - (new_idx-CONTEXT)] = self.training_data[i][0]
        return_tensor = torch.tensor(return_matrix)
        return_label = torch.tensor(np.array(self.training_data[new_idx][1]), dtype=torch.long)
        return return_tensor, return_label


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (frame, label) in enumerate(train_loader):
        frame, label = frame.to(device), label.to(device)
        optimizer.zero_grad()
        out_put = model(frame.float())
        loss = torch.nn.functional.nll_loss(out_put, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('iteration: {}, training_loss: {}'.format(batch_idx, loss.item()))


def valid(model, device, valid_loader):
    model.eval()
    valid_loss = 0
    right_cnt = 0
    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            out_put = model(data.float())
            valid_loss += torch.nn.functional.nll_loss(out_put, label)
            res_label = out_put.argmax(dim=1, keepdim=True)
            right_cnt += res_label.eq(label.view_as(res_label)).sum().item()
    print('validate_accuracy: {}'.format(right_cnt / len(valid_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    res_label = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out_put = model(data.float())
            res = out_put.argmax(dim=1, keepdim=False)
            # if res.data.cpu().item() != 0:
            res_label.append(res.data.cpu().item())

    return res_label


def main():
    # 1. Load input data.
    # train_data = np.load('train.npy', allow_pickle=True)
    # train_label = np.load('train_labels.npy', allow_pickle=True)
    train_data_set = MyDataset(data_npy_file='train.npy', label_npy_file='train_labels.npy')
    print('training data set is ready.')
    # valid_data = np.load('dev.npy', allow_pickle=True)
    # valid_label = np.load('dev_labels.npy', allow_pickle=True)
    validate_data_set = MyDataset(data_npy_file='dev.npy', label_npy_file='dev_labels.npy')
    print('validate data set is ready.')
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=NUM_WORKERS)
    validate_loader = torch.utils.data.DataLoader(validate_data_set, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=NUM_WORKERS)

    print('data loader is ready.')

    device = torch.device("cuda")

    # 2. Init model.
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 3. Train and validate.
    for epoch in range(1, 1+EPOCH):
        train(model, device, train_loader, optimizer)
        print('finish training.')
        valid(model, device, validate_loader)
        print('finish validate.')
        scheduler.step()

    # 4. Get res label.
    test_data = np.load('test.npy', allow_pickle=True)
    cur_idx = 0
    print('test data is loaded.')
    with open(OUTPUT_PATH, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id', 'label'])
        for idx in range(test_data.shape[0]):
            clip = test_data[idx]
            print('clip shape:', clip.shape)
            test_data_set = TestDataset(clip)
            test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=1, shuffle=False,
                                                      num_workers=NUM_WORKERS)
            res_label = test(model, device, test_loader)
            for label in res_label:
                writer.writerow([str(cur_idx), str(label)])
                cur_idx += 1
            print('finish one clip.')
        print('finish predict')


if __name__ == '__main__':
    main()
