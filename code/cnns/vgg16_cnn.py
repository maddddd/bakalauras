import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
import numpy as np
import pics_dataset
import tools


class VGG16_CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, data_set_type, image_size=64):
        super(VGG16_CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = 2
        self.loss_history = []
        self.acc_history = []
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')
        self.conv1 = nn.Conv2d(1, 4, 3)  # grayscale, 8 conv. filters, 3x3 size
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 3)
        self.bn2 = nn.BatchNorm2d(4)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(4, 4, 3)
        self.bn3 = nn.BatchNorm2d(4)
        self.conv4 = nn.Conv2d(4, 4, 3)
        self.bn4 = nn.BatchNorm2d(4)
        self.conv5 = nn.Conv2d(4, 4, 3)
        self.bn5 = nn.BatchNorm2d(4)
        self.conv6 = nn.Conv2d(4, 4, 3)
        self.bn6 = nn.BatchNorm2d(4)
        self.conv7 = nn.Conv2d(4, 4, 3)
        self.bn7 = nn.BatchNorm2d(4)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv8 = nn.Conv2d(4, 4, 3)
        self.bn8 = nn.BatchNorm2d(4)
        self.conv9 = nn.Conv2d(4, 4, 3)
        self.bn9 = nn.BatchNorm2d(4)
        self.conv10 = nn.Conv2d(4, 4, 3)
        self.bn10 = nn.BatchNorm2d(4)
        # dilated convolutions
        self.dil_conv1 = nn.Conv2d(4, 8, 3, dilation=3)
        self.bn11 = nn.BatchNorm2d(8)
        self.dil_conv2 = nn.Conv2d(8, 4, 3, dilation=2)
        self.bn12 = nn.BatchNorm2d(4)

        self.input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(self.input_dims, self.num_classes)  # perceptron
        print(self.fc1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # for learning
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.data_set = pics_dataset.LungsDataSet(data_set_type, self.batch_size, 64)
        self.data_loader = self.data_set.data_loader

    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, self.image_size, self.image_size))

        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.maxpool1(batch_data)
        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn = resized_batch_data[:, :, 24:40, 24:40].flatten()

        batch_data = self.conv3(batch_data)
        batch_data = self.conv4(batch_data)
        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.conv7(batch_data)
        batch_data = self.maxpool2(batch_data)
        batch_data = self.conv8(batch_data)
        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.conv9(batch_data)
        batch_data = self.conv10(batch_data)
        batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.dil_conv1(batch_data)
        batch_data = self.dil_conv2(batch_data)
        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        return int(np.prod(hypercolumn.size()))

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
        batch_data = self.maxpool1(batch_data)

        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn = resized_batch_data[:, :, 24:40, 24:40].flatten()

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.conv7(batch_data)
        batch_data = self.bn7(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.maxpool2(batch_data)

        batch_data = self.conv8(batch_data)
        batch_data = self.bn8(batch_data)
        batch_data = F.relu(batch_data)

        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.conv9(batch_data)
        batch_data = self.bn9(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv10(batch_data)
        batch_data = self.bn10(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)

        batch_data = self.dil_conv1(batch_data)
        batch_data = self.bn11(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.dil_conv2(batch_data)
        batch_data = self.bn12(batch_data)
        batch_data = F.relu(batch_data)

        resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=False)
        hypercolumn_read = resized_batch_data[:, :, 24:40, 24:40].flatten()
        hypercolumn = torch.cat((hypercolumn, hypercolumn_read), dim=0)
        hypercolumn = hypercolumn.view(self.batch_size, -1)
        print(hypercolumn.size())
        classes = self.fc1(hypercolumn)

        return classes

    def train_vgg16_cnn(self):
        self.train(mode=True)
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for j, (pic, label) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device)
                pic = pic.to(self.device)
                if pic.size()[0] == self.batch_size:
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

    """
    def test_vg16_cnn(self):
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
    """


if __name__ == "__main__":
    vgg16 = VGG16_CNN(0.0001, 500, 48, 'untouched')
    vgg16.train_vgg16_cnn()
    #tools.save_model(vgg16, 'vgg16_cnn')

    """
    cnn.train_cnn()
    tools.save_model(cnn, 'or_cnn')
    load_path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets', 'or_cnn.pt'))
    cnn_2 = tools.load_or_cnn_model(0.001, 5, 48, 40, load_path)
    cnn_2.test_cnn()
    """
