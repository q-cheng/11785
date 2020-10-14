import csv

import torch
from torch import norm
from torch.nn import Conv2d, BatchNorm2d, Dropout, ReLU, Sequential, MaxPool2d, Linear, CrossEntropyLoss, Module, \
    Dropout2d, CosineSimilarity
from torch.nn.functional import softmax
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
# from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip
import numpy as np
from PIL import Image

# Global Variable

BATCH_SIZE = 100
EPOCH = 15
NUM_OF_WORKER = 4
# LEARNING_RATE = 0.15
TRAIN_PATH = 'classification_data/train_data/'
VALID_PATH = 'classification_data/valid_data/'
TEST_PATH = 'classification_data/test_data/'
VERI_TEST_PATH = 'verification_pairs_test.txt'
OUTPUT_PATH = 'res.csv'


def init_weights(m):
    if type(m) == Conv2d or type(m) == Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = VGGBlock(3, 64, 3, 1, 1, 0.3)
        self.layer2 = VGGBlock(64, 64, 3, 1, 1)
        self.layer3 = VGGBlock(64, 64, 3, 1, 1)
        self.layer4 = VGGBlock(64, 128, 3, 1, 1, 0.3)
        self.layer5 = VGGBlock(128, 128, 3, 1, 1)
        self.layer6 = VGGBlock(128, 256, 3, 1, 1, 0.3)
        self.layer7 = VGGBlock(256, 256, 3, 1, 1)
        self.layer8 = VGGBlock(256, 256, 3, 1, 1)
        self.layer9 = VGGBlock(256, 256, 3, 1, 1)
        self.layer10 = VGGBlock(256, 256, 3, 1, 1)
        self.max_pool = MaxPool2d(2, 2)
        self.linear_layer = Linear(256, 4000, bias=False)
        self.layers = [self.layer1, self.layer2, self.max_pool, self.layer3, self.max_pool, self.layer4, self.layer5,
                       self.max_pool, self.layer6, self.layer7, self.max_pool, self.layer8, self.max_pool,
                       self.layer9, self.layer10, self.max_pool]
        self.layers = Sequential(*self.layers)

    def forward(self, x):
        out = x
        out = self.layers(out)  # Embedding vector.
        embedding = torch.flatten(out, 1)
        linear_out = self.linear_layer(embedding)  # vector -> (1 * 4000)
        return linear_out / norm(self.linear_layer.weight, dim=1), embedding


class VGGBlock(Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dropout=0):
        super(VGGBlock, self).__init__()
        self.conv_layer = Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                                 stride=stride, padding=padding)
        self.bn = BatchNorm2d(out_channel)
        self.ReLu = ReLU(inplace=True)
        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.bn(out)
        out = self.ReLu(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class MyDataset(Dataset):

    def __init__(self, test_files):
        self.test_pairs = []
        for test_pair in test_files:
            # print(test_pair)
            cur_line = test_pair.split(' ')  # Two file name.
            self.test_pairs.append((cur_line[0], cur_line[1]))

    def __len__(self):
        return len(self.test_pairs)

    def __getitem__(self, idx):
        img_1 = Image.open(self.test_pairs[idx][0])
        img_1 = ToTensor()(img_1)
        img_1 = RandomHorizontalFlip()(img_1)
        img_1 = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(img_1)
        img_2 = Image.open(self.test_pairs[idx][1])
        img_2 = ToTensor()(img_2)
        img_2 = RandomHorizontalFlip()(img_2)
        img_2 = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(img_2)
        return img_1, img_2


def load_data(train_path, valid_path, test_path):
    transform = Compose([ToTensor(), RandomHorizontalFlip(), Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
    train_loader = DataLoader(ImageFolder(train_path, transform=transform), batch_size=BATCH_SIZE,
                              num_workers=NUM_OF_WORKER, shuffle=True)
    # valid_loader = DataLoader(ImageFolder(valid_path, transform=ToTensor()), batch_size=BATCH_SIZE,
    # num_workers=NUM_OF_WORKER)
    test_loader = DataLoader(ImageFolder(test_path, transform=transform), batch_size=BATCH_SIZE,
                             num_workers=NUM_OF_WORKER, shuffle=True)
    return train_loader, None, test_loader


def train(model, data_loader, test_loader, device, optimizer, scheduler, criterion, task='Classification'):
    model.train()

    for epoch in range(EPOCH):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)[0]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1, batch_num + 1, avg_loss / 50))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader, device, criterion)
            train_loss, train_acc = test_classify(model, data_loader, device, criterion)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
            torch.save(model, 'my_model/model_{}'.format(epoch))
            print('model saved')
        # else:
        #     test_verify(model, test_loader, device, criterion)
        scheduler.step()


def test_classify(model, test_loader, device, criterion):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0
    with torch.no_grad():
        for batch_num, (feats, labels) in enumerate(test_loader):
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)[0]

            _, pred_labels = torch.max(softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            loss = criterion(outputs, labels.long())

            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()] * feats.size()[0])
            del feats
            del labels

    model.train()
    return np.mean(test_loss), accuracy / total


def test_verify(model, test_loader, device):
    model.eval()
    cos = CosineSimilarity(dim=1, eps=1e-6)
    similarity_list = []
    with torch.no_grad():
        for batch_num, (img1, img2) in enumerate(test_loader):
            print('cur:', batch_num, 'total:', len(test_loader))
            img1, img2 = img1.to(device), img2.to(device)
            embedding1, embedding2 = model(img1)[1], model(img2)[1]
            similarity = cos(embedding1, embedding2)
            similarity_list.append(similarity.cpu().item())
    return similarity_list


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    print('Start loading data.')
    train_loader, _, test_loader = load_data(TRAIN_PATH, VALID_PATH, TEST_PATH)
    print('Finish loading data.')
    model = Net()
    model.to(device)
    # optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-5)
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
    criterion = CrossEntropyLoss()
    print('Start training.')
    train(model, train_loader, test_loader, device, optimizer, scheduler, criterion, 'Classification')
    print('Finish training')

    #     model = torch.load('my_model/model_10')
    #     model.to(device)
    #     # file_list = []
    #     print('Start load file.')
    #     with open(VERI_TEST_PATH) as in_put:
    #         file_list = in_put.read()
    #     file_list = file_list.split('\n')
    #     # print(file_list)
    #     file_list = file_list[:-1]
    #
    #     print('Finish load file.')
    #     predict_loader = DataLoader(MyDataset(file_list), batch_size=1, shuffle=False)
    #     predict = test_verify(model, predict_loader, device)
    #     print('Start predict.')
    #     with open(OUTPUT_PATH, 'w') as out_put:
    #         writer = csv.writer(out_put, delimiter=',')
    #         writer.writerow(['id', 'category'])
    #         for idx, file in enumerate(file_list):
    #             print('cur:', idx, 'total:', len(file_list))
    #             # print(file_list[idx]+','+str(predict[idx]))
    #             writer.writerow([file_list[idx], str(predict[idx])])
    #     print('Finish predict.')
    #     # print(len(file_list))
    return


if __name__ == '__main__':
    main()
