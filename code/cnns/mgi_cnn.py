import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
import numpy as np
import pics_dataset
import os
from pathlib import Path


class MGI_CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, data_set_type):
        super(MGI_CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = 2
        self.loss_history = []
        self.acc_history = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
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
        self.data_set = pics_dataset.LungsDataSet(data_set_type, self.batch_size)
        self.tensor_data_set = self.data_set.tensor_data_set
        self.data_loader = self.data_set.data_loader