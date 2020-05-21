import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
import numpy as np
import pics_dataset
import tools
import os
from pathlib import Path


class CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, data_set_type, image_size):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = 2
        self.loss_history = []
        self.acc_history = []
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.conv1 = nn.Conv2d(1, 32, 3)  # grayscale, 32 conv. filters, 3x3 size
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(input_dims, self.num_classes)  # perceptron
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # for learning
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.data_set = pics_dataset.LungsDataSet(data_set_type, self.batch_size, self.image_size)    # resize to 40x40 pixels
        self.tensor_data_set = self.data_set.tensor_data_set
        self.data_loader = self.data_set.data_loader
        self.temp = 0

    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, self.image_size, self.image_size))
        batch_data = self.conv1(batch_data)
        # batch_data = self.bn1(batch_data)
        batch_data = self.conv2(batch_data)
        # batch_data = self.bn2(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size()))

    def forward_pass(self, batch_data):
        batch_data = batch_data.unsqueeze(dim=1)
        batch_data = batch_data.type(torch.float) / 255
        batch_data = batch_data.to(self.device, dtype=torch.float)
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)
        batch_data = batch_data.view(batch_data.size()[0], -1)

        classes = self.fc1(batch_data)

        return classes

    def train_cnn(self):
        self.train(mode=True)
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for j, (pic, label) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device)
                pic = pic.to(self.device)
                prediction = self.forward_pass(pic)
                prediction = prediction.to(self.device)
                loss = self.loss(prediction, label)
                prediction = F.softmax(prediction, dim=0)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Finish epoch ', i, 'epoch loss %.3f ' % ep_loss,
                  'accuracy %.3f ' % np.mean(ep_acc))
            self.loss_history.append(ep_loss)

    def test_cnn(self):
        self.train(mode=False)
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for j, (pic, label) in enumerate(self.data_loader):
                label = label.to(self.device)
                pic = pic.to(self.device)
                prediction = self.forward_pass(pic)
                prediction = prediction.to(self.device)
                loss = self.loss(prediction, label)
                prediction = F.softmax(prediction, dim=0)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
            print('Finish epoch ', i, 'epoch loss %.3f ' % ep_loss,
                  'accuracy %.3f ' % np.mean(ep_acc))
            self.loss_history.append(ep_loss)


if __name__ == "__main__":
    cnn = CNN(0.001, 50, 48, 'untouched', 40)
    cnn.train_cnn()
    tools.save_model(cnn, 'or_cnn')
    load_path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets', 'or_cnn.pt'))
    cnn_2 = tools.load_or_cnn_model(0.001, 5, 48, 40, load_path)
    cnn_2.test_cnn()
