import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
import numpy as np
from PIL import Image
import pics_dataset
import tools
import os
from pathlib import Path


# from numpy array -> numpy array
def downsize_image(image, new_size):
    img = Image.fromarray(image)
    img = img.resize((new_size, new_size), resample=Image.NEAREST)
    img = np.array(img)
    return img


def get_resized_images(images):
    pics_20x20 = []
    pics_30x30 = []
    pics_40x40 = []
    for image in images:
        img = image.numpy()
        pics_20x20.append(pics_dataset.cut_image(img, 20))
        pics_30x30.append(downsize_image(pics_dataset.cut_image(img, 30), 20))
        pics_40x40.append(downsize_image(pics_dataset.cut_image(img, 40), 20))
    pics_20x20 = torch.tensor(pics_20x20)
    pics_30x30 = torch.tensor(pics_30x30)
    pics_40x40 = torch.tensor(pics_40x40)
    return pics_20x20, pics_30x30, pics_40x40


class MGI_CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, data_set_type):
        super(MGI_CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = 2
        self.loss_history = []
        self.acc_history = []
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')

        # zoom-in
        self.conv1_zoom_in = nn.Conv2d(1, 16, 3, padding=(1, 1))  # grayscale, 16 conv. filters, 3x3 size
        self.bn1_zoom_in = nn.BatchNorm2d(16)
        self.conv2_zoom_in = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.bn2_zoom_in = nn.BatchNorm2d(16)
        self.maxpool1_in = nn.MaxPool2d(2, stride=1)
        self.conv3_zoom_in = nn.Conv2d(1, 16, 3, padding=(1, 1))
        self.bn3_zoom_in = nn.BatchNorm2d(16)
        self.conv4_zoom_in = nn.Conv2d(32, 16, 3, padding=(1, 1))
        self.bn4_zoom_in = nn.BatchNorm2d(16)
        self.maxpool2_in = nn.MaxPool2d(2, stride=1)
        self.conv5_zoom_in = nn.Conv2d(1, 16, 3, padding=(1, 1))
        self.bn5_zoom_in = nn.BatchNorm2d(16)
        self.conv6_zoom_in = nn.Conv2d(32, 16, 3, padding=(1, 1))
        self.bn6_zoom_in = nn.BatchNorm2d(16)
        self.maxpool3_in = nn.MaxPool2d(2, stride=1)

        # zoom-out

        self.conv1_zoom_out = nn.Conv2d(1, 16, 3, padding=(1, 1))  # grayscale, 16 conv. filters, 3x3 size
        self.bn1_zoom_out = nn.BatchNorm2d(16)
        self.conv2_zoom_out = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.bn2_zoom_out = nn.BatchNorm2d(16)
        self.maxpool1_out = nn.MaxPool2d(2, stride=1)
        self.conv3_zoom_out = nn.Conv2d(1, 16, 3, padding=(1, 1))
        self.bn3_zoom_out = nn.BatchNorm2d(16)
        self.conv4_zoom_out = nn.Conv2d(32, 16, 3, padding=(1, 1))
        self.bn4_zoom_out = nn.BatchNorm2d(16)
        self.maxpool2_out = nn.MaxPool2d(2, stride=1)
        self.conv5_zoom_out = nn.Conv2d(1, 16, 3, padding=(1, 1))
        self.bn5_zoom_out = nn.BatchNorm2d(16)
        self.conv6_zoom_out = nn.Conv2d(32, 16, 3, padding=(1, 1))
        self.bn6_zoom_out = nn.BatchNorm2d(16)
        self.maxpool3_out = nn.MaxPool2d(2, stride=1)

        # zoom-in and zoom-out combined

        self.conv1 = nn.Conv2d(32, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2, stride=1)

        input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(input_dims, self.num_classes)  # perceptron
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # for learning
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

        self.data_set = pics_dataset.LungsDataSet(data_set_type, self.batch_size, 40)
        self.data_loader = self.data_set.data_loader
        """
        self.data_set_20 = pics_dataset.LungsDataSet(data_set_type, self.batch_size, 20, True, downsize_dim)
        self.tensor_data_set_20 = self.data_set_20.tensor_data_set
        self.data_loader_20 = self.data_set_20.data_loader

        # cut out 30x30 pixel images, downsized to downsized_dim x downsized_dim pixels
        self.data_set_30 = pics_dataset.LungsDataSet(data_set_type, self.batch_size, 30, True, downsize_dim)
        self.tensor_data_set_30 = self.data_set_30.tensor_data_set
        self.data_loader_30 = self.data_set_30.data_loader

        # cut out 40x40 pixel images, downsized to downsized_dim x downsized_dim pixels
        self.data_set_40 = pics_dataset.LungsDataSet(data_set_type, self.batch_size, True, downsize_dim)
        self.tensor_data_set_40 = self.data_set_40.tensor_data_set
        self.data_loader_40 = self.data_set_40.data_loader
        """

    def calc_input_dims(self):
        batch_data_40 = T.zeros((1, 1, 20, 20))
        batch_data_30 = T.zeros((1, 1, 20, 20))
        batch_data_20 = T.zeros((1, 1, 20, 20))

        # ZOOM IN

        # get filters for batch_data_40 and train them
        batch_data_40_in = self.conv1_zoom_in(batch_data_40)
        batch_data_40_in = self.bn1_zoom_in(batch_data_40_in)
        batch_data_40_in = F.relu(batch_data_40_in)
        batch_data_zoom_in = self.conv2_zoom_in(batch_data_40_in)
        batch_data_zoom_in = self.bn2_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = F.relu(batch_data_zoom_in)
        batch_data_zoom_in = self.maxpool1_in(batch_data_zoom_in)
        batch_data_zoom_in = F.pad(batch_data_zoom_in, [0, 1, 0, 1], mode='replicate')

        # get filters for batch_data_30
        batch_data_30_in = self.conv3_zoom_in(batch_data_30)
        batch_data_30_in = self.bn3_zoom_in(batch_data_30_in)
        batch_data_30_in = F.relu(batch_data_30_in)

        # concatenate filters trained on batch_data_40 and batch_data_30 and train them further on batch_data_30
        batch_data_zoom_in = torch.cat((batch_data_zoom_in, batch_data_30_in), dim=1)
        batch_data_zoom_in = self.conv4_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = self.bn4_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = F.relu(batch_data_zoom_in)
        batch_data_zoom_in = self.maxpool2_in(batch_data_zoom_in)
        batch_data_zoom_in = F.pad(batch_data_zoom_in, [0, 1, 0, 1], mode='replicate')

        # get filters for batch_data_20
        batch_data_20_in = self.conv5_zoom_in(batch_data_20)
        batch_data_20_in = self.bn5_zoom_in(batch_data_20_in)
        batch_data_20_in = F.relu(batch_data_20_in)

        # concatenate combined batch_data_40 and batch_data_30 filters and batch_data_20 filters
        # and train them further on batch_data_20
        batch_data_zoom_in = torch.cat((batch_data_zoom_in, batch_data_20_in), dim=1)
        batch_data_zoom_in = self.conv6_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = self.bn6_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = F.relu(batch_data_zoom_in)
        batch_data_zoom_in = self.maxpool3_in(batch_data_zoom_in)
        batch_data_zoom_in = F.pad(batch_data_zoom_in, [0, 1, 0, 1], mode='replicate')

        # ZOOM OUT

        batch_data_zoom_out = batch_data_zoom_in    # same size

        # COMBINED

        # concat zoom in and zoom out features:
        batch_data = torch.cat((batch_data_zoom_in, batch_data_zoom_out), dim=1)
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.maxpool1(batch_data)

        return int(np.prod(batch_data.size()))

    def forward_pass(self, batch_data_20, batch_data_30, batch_data_40):
        batch_data_40 = batch_data_40.unsqueeze(dim=1)
        batch_data_40 = batch_data_40.type(torch.float) / 255
        batch_data_40 = batch_data_40.to(self.device, dtype=torch.float)
        batch_data_30 = batch_data_30.unsqueeze(dim=1)
        batch_data_30 = batch_data_30.type(torch.float) / 255
        batch_data_30 = batch_data_30.to(self.device, dtype=torch.float)
        batch_data_20 = batch_data_20.unsqueeze(dim=1)
        batch_data_20 = batch_data_20.type(torch.float) / 255
        batch_data_20 = batch_data_20.to(self.device, dtype=torch.float)

        # ZOOM IN

        # get filters for batch_data_40 and train them
        batch_data_40_in = self.conv1_zoom_in(batch_data_40)
        batch_data_40_in = self.bn1_zoom_in(batch_data_40_in)
        batch_data_40_in = F.relu(batch_data_40_in)
        batch_data_zoom_in = self.conv2_zoom_in(batch_data_40_in)
        batch_data_zoom_in = self.bn2_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = F.relu(batch_data_zoom_in)
        batch_data_zoom_in = self.maxpool1_in(batch_data_zoom_in)
        batch_data_zoom_in = F.pad(batch_data_zoom_in, [0, 1, 0, 1], mode='replicate')

        # get filters for batch_data_30
        batch_data_30_in = self.conv3_zoom_in(batch_data_30)
        batch_data_30_in = self.bn3_zoom_in(batch_data_30_in)
        batch_data_30_in = F.relu(batch_data_30_in)

        # concatenate filters trained on batch_data_40 and batch_data_30 and train them further on batch_data_30
        batch_data_zoom_in = torch.cat((batch_data_zoom_in, batch_data_30_in), dim=1)
        batch_data_zoom_in = self.conv4_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = self.bn4_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = F.relu(batch_data_zoom_in)
        batch_data_zoom_in = self.maxpool2_in(batch_data_zoom_in)
        batch_data_zoom_in = F.pad(batch_data_zoom_in, [0, 1, 0, 1], mode='replicate')

        # get filters for batch_data_20
        batch_data_20_in = self.conv5_zoom_in(batch_data_20)
        batch_data_20_in = self.bn5_zoom_in(batch_data_20_in)
        batch_data_20_in = F.relu(batch_data_20_in)

        # concatenate combined batch_data_40 and batch_data_30 filters and batch_data_20 filters
        # and train them further on batch_data_20
        batch_data_zoom_in = torch.cat((batch_data_zoom_in, batch_data_20_in), dim=1)
        batch_data_zoom_in = self.conv6_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = self.bn6_zoom_in(batch_data_zoom_in)
        batch_data_zoom_in = F.relu(batch_data_zoom_in)
        batch_data_zoom_in = self.maxpool3_in(batch_data_zoom_in)
        batch_data_zoom_in = F.pad(batch_data_zoom_in, [0, 1, 0, 1], mode='replicate')

        # ZOOM OUT

        # get filters for batch_data_20 and train them
        batch_data_20_out = self.conv1_zoom_out(batch_data_20)
        batch_data_20_out = self.bn1_zoom_out(batch_data_20_out)
        batch_data_20_out = F.relu(batch_data_20_out)
        batch_data_zoom_out = self.conv2_zoom_out(batch_data_20_out)
        batch_data_zoom_out = self.bn2_zoom_out(batch_data_zoom_out)
        batch_data_zoom_out = F.relu(batch_data_zoom_out)
        batch_data_zoom_out = self.maxpool1_out(batch_data_zoom_out)
        batch_data_zoom_out = F.pad(batch_data_zoom_out, [0, 1, 0, 1], mode='replicate')

        # get filters for batch_data_30
        batch_data_30_out = self.conv3_zoom_out(batch_data_30)
        batch_data_30_out = self.bn3_zoom_out(batch_data_30_out)
        batch_data_30_out = F.relu(batch_data_30_out)

        # concatenate filters trained on batch_data_20 and batch_data_30 and train them further on batch_data_30
        batch_data_zoom_out = torch.cat((batch_data_zoom_out, batch_data_30_out), dim=1)
        batch_data_zoom_out = self.conv4_zoom_in(batch_data_zoom_out)
        batch_data_zoom_out = self.bn4_zoom_in(batch_data_zoom_out)
        batch_data_zoom_out = F.relu(batch_data_zoom_out)
        batch_data_zoom_out = self.maxpool2_out(batch_data_zoom_out)
        batch_data_zoom_out = F.pad(batch_data_zoom_out, [0, 1, 0, 1], mode='replicate')

        # get filters for batch_data_40
        batch_data_40_out = self.conv5_zoom_out(batch_data_40)
        batch_data_40_out = self.bn5_zoom_out(batch_data_40_out)
        batch_data_40_out = F.relu(batch_data_40_out)

        # concatenate combined batch_data_20 and batch_data_30 filters and batch_data_40 filters
        # and train them further on batch_data_40
        batch_data_zoom_out = torch.cat((batch_data_zoom_out, batch_data_40_out), dim=1)
        batch_data_zoom_out = self.conv6_zoom_in(batch_data_zoom_out)
        batch_data_zoom_out = self.bn6_zoom_in(batch_data_zoom_out)
        batch_data_zoom_out = F.relu(batch_data_zoom_out)
        batch_data_zoom_out = self.maxpool3_out(batch_data_zoom_out)
        batch_data_zoom_out = F.pad(batch_data_zoom_out, [0, 1, 0, 1], mode='replicate')

        # COMBINED:

        # concat zoom in and zoom out features:
        batch_data = torch.cat((batch_data_zoom_in, batch_data_zoom_out), dim=1)
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = batch_data.view(batch_data.size()[0], -1)

        classes = self.fc1(batch_data)

        return classes

    def train_mgi_cnn(self):
        self.train(mode=True)
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for j, (pics, labels) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                labels = labels.to(self.device)
                pics_20x20, pics_30x30, pics_40x40 = get_resized_images(pics)
                pics_20x20 = pics_20x20.to(self.device)
                pics_30x30 = pics_30x30.to(self.device)
                pics_40x40 = pics_40x40.to(self.device)
                prediction = self.forward_pass(pics_20x20, pics_30x30, pics_40x40)
                prediction = prediction.to(self.device)
                loss = self.loss(prediction, labels)
                prediction = F.softmax(prediction, dim=0)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != labels,
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


if __name__ == "__main__":
    mgi = MGI_CNN(0.001, 10, 48, 'untouched')
    mgi.train_mgi_cnn()
    tools.save_model(mgi, 'mgi_cnn')
    load_path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets', 'mgi_cnn.pt'))
    mgi_2 = tools.load_mgi_cnn_model()
