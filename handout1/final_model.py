import numpy as np
import torch
import csv

# Constants
CONTEXT = 50  # 30
EPOCH = 4
BATCH_SIZE = 1024  # 512
NUM_WORKERS = 4
LEARNING_RATE = 0.01
OUTPUT_PATH = 'res.csv'


# MLP model
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(((1 + 2 * CONTEXT) * 13), 4096)  # Hidden layer.
        self.bn1 = torch.nn.BatchNorm1d(num_features=4096)
        self.dropout1 = torch.nn.Dropout(0.02)
        self.layer2 = torch.nn.Linear(4096, 2048)
        self.bn2 = torch.nn.BatchNorm1d(num_features=2048)
        self.dropout2 = torch.nn.Dropout(0.01)
        self.layer3 = torch.nn.Linear(2048, 1024)
        self.bn3 = torch.nn.BatchNorm1d(num_features=1024)
        self.dropout3 = torch.nn.Dropout(0.01)
        self.layer4 = torch.nn.Linear(1024, 512)
        self.bn4 = torch.nn.BatchNorm1d(num_features=512)
        # self.dropout4 = torch.nn.Dropout(0.01)
        self.layer5 = torch.nn.Linear(512, 346)  # Output layer.
        self.reLu = torch.nn.functional.relu
        # self.reLu = torch.nn.functional.leaky_relu

    def forward(self, x):
        x = x.view((-1, (1 + 2 * CONTEXT) * 13))
        x = self.dropout1(x)
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.reLu(x)
        x = self.dropout2(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.reLu(x)
        x = self.dropout3(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.reLu(x)
        # x = self.dropout4(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.reLu(x)
        x = self.layer5(x)
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
        input_data = test_data
        self.input_after_pad = pad_central(input_data)

    def __len__(self):
        return self.input_after_pad.shape[0] - 2 * CONTEXT

    def __getitem__(self, idx):
        new_idx = idx + CONTEXT
        return_matrix = np.array(self.input_after_pad[new_idx - CONTEXT:new_idx + CONTEXT + 1]).reshape(-1)
        return_tensor = torch.tensor(return_matrix, dtype=torch.float)
        return return_tensor


# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_npy_file, label_npy_file, transform=None):
        input_data = np.load(data_npy_file, allow_pickle=True)
        input_label = np.load(label_npy_file, allow_pickle=True)
        self.training_data = []
        self.training_label = []
        # Padding with CONTEXT number of np.zeros([Y x Z]) and Flattern the frames.
        for group_idx in range(input_data.shape[0]):
            cur_group, cur_label = input_data[group_idx], input_label[group_idx]
            group_after_pad = pad_central(cur_group)
            label_after_pad = pad_central(cur_label)
            for group in group_after_pad:
                self.training_data.append(group)
            for label in label_after_pad:
                self.training_label.append(label)
        # print('train_length:', len(self.training_data), 'label_length:', len(self.training_label))

    def __len__(self):
        # print('length:', len(self.training_data) - 2 * CONTEXT)
        return len(self.training_data) - 2 * CONTEXT

    def __getitem__(self, idx):
        new_idx = idx + CONTEXT  # Start idx in training data is CONTEXT.
        # Train and validate.
        return_matrix = np.array(self.training_data[new_idx-CONTEXT:new_idx+CONTEXT+1]).reshape(-1)
        return_tensor = torch.tensor(return_matrix, dtype=torch.float)
        return_label = torch.tensor(np.array(self.training_label[new_idx]).reshape(-1), dtype=torch.long)
        return return_tensor, return_label


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        out_put = model(data)
        # out_put = torch.argmax(out_put, 1, keepdim=True)
        label = label.flatten()
        # print(out_put.shape, label.shape)
        loss = criterion(out_put, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('iteration: {}, training_loss: {}'.format(batch_idx, loss.item()))


def valid(model, device, valid_loader):
    model.eval()
    right_cnt = 0
    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            out_put = model(data)
            res_label = out_put.argmax(dim=1, keepdim=True)
            right_cnt += res_label.eq(label.view_as(res_label)).sum().item()
    print('validate_accuracy: {}'.format(right_cnt / len(valid_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    res_label = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out_put = model(data)
            res = out_put.data.argmax(dim=1, keepdim=False)
            # print('predict:', res)
            res_label.append(res.data.cpu().item())

    return res_label


def main():
    # 1. Load input data.
    train_data_set = MyDataset(data_npy_file='train.npy', label_npy_file='train_labels.npy')
    print('training data set is ready.')
    validate_data_set = MyDataset(data_npy_file='dev.npy', label_npy_file='dev_labels.npy')
    print('validate data set is ready.')
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=NUM_WORKERS)
    validate_loader = torch.utils.data.DataLoader(validate_data_set, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=NUM_WORKERS)
    print('data loader is ready.')

    device = torch.device("cuda")

    # 2. Init model.
    bias_p = weight_p = []
    model = MLP().to(device)

    weight_list = [model.layer1.weight] + [model.bn1.weight] + [model.layer2.weight] + [model.bn2.weight] +\
                  [model.layer3.weight] + [model.bn3.weight] + [model.layer4.weight]+ [model.bn4.weight] +\
                  [model.layer5.weight]

    bias_list = [model.layer1.bias] + [model.bn1.bias] + [model.layer2.bias] + [model.bn2.bias] + \
                  [model.layer3.bias] + [model.bn3.bias] + [model.layer4.bias] + [model.bn4.bias] + \
                  [model.layer5.bias]

    optimizer = torch.optim.Adam([{'params': weight_list, 'weight_decay': 1e-5},
          {'params': bias_list, 'weight_decay': 1e-11}])
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-05)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-11)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    criterion = torch.nn.CrossEntropyLoss()
    # 3. Train and validate.
    for epoch in range(1, 1+EPOCH):
        train(model, device, train_loader, optimizer, criterion)
        print('finish training.')
        # print('validate with validation set:')
        valid(model, device, validate_loader)
        # print('get training accuracy:')
        # valid(model, device, train_loader)
        print('finish validate.')
        scheduler.step()
        # optimizer.step()

    # 4. Get res label.
    test_data = np.load('test.npy', allow_pickle=True)
    cur_idx = 0
    print('test data is loaded.')
    with open(OUTPUT_PATH, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id', 'label'])
        for idx in range(test_data.shape[0]):
            clip = test_data[idx]
            print('cnt:', cur_idx)
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
