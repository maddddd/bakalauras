import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
import numpy as np
import pics_dataset
import tools

"""
    "Paprastas" konvoliucinis neuroninis tinklas.
"""


class CNN(nn.Module):
    """
        lr              - mokymosi greitis
        epochs          - epochu skaicius
        batch_size      - mokymosi parametras, nusakantis kas kiek nuotrauku turi buti atnaujinti svoriai.
        data_set_type   - 'train', 'test', 'untouched', 'none'
        image_size      - paveikslo dydis. Negali buti didesnis nei realaus paveikslo diske dydis
                        (testineje nuotrauku kolekcijoje - 65x65 pikseliai). Jei nurodytas dydis mazesnis,
                        nuotrauka apkerpama.
    """
    def __init__(self, lr, epochs, batch_size, data_set_type, image_size):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.data_set_type = data_set_type
        self.image_size = image_size
        self.num_classes = 2
        self.loss_history = []
        self.acc_history = []
        self.false_pos_history = []
        self.false_neg_history = []
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
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # mokymuisi!
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.data_set = None
        self.data_loader = None
        if data_set_type != 'none':
            self.data_set = pics_dataset.LungsDataSet(data_set_type, self.batch_size, self.image_size)
            self.data_loader = self.data_set.data_loader

    """
        funkcija, skirta nustatyti paskutinio tinklo sluoksnio - perceptrono - ieiciu kiekiui. 
        Per tinkla leidziamas testinis tenzorius, o jo galutinis dydis nusako, koks turi buti perceptronas.
    """
    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, self.image_size, self.image_size))
        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size()))

    """
        skleidimo pirmyn funkcija.
    """
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

    """
        treniravimo funkcija.
    """
    def train_cnn(self):
        if self.data_set_type == 'none':
            print('Nuotraukos nebuvo ikeltos (pasirinkite kitoki tinklo tipa)')
            return
        self.train(mode=True)
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            ep_false_pos = []
            ep_false_neg = []
            for j, (pics, labels) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                labels = labels.to(self.device)
                pics = pics.to(self.device)
                prediction = self.forward_pass(pics)
                prediction = prediction.to(self.device)
                loss = self.loss(prediction, labels)
                prediction = F.softmax(prediction, dim=0)
                classes = T.argmax(prediction, dim=1)

                wrong = T.where(classes != labels,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_acc.append(acc.item())
                ep_loss += loss.item()

                false_pos = []
                false_neg = []
                for k in range(0, len(classes)):
                    if labels[k] == 0 and classes[k] == 1:
                        false_pos.append(1.)
                    else:
                        false_pos.append(0.)
                    if labels[k] == 1 and classes[k] == 0:
                        false_neg.append(1.)
                    else:
                        false_neg.append(0.)
                false_pos = np.array(false_pos)
                false_neg = np.array(false_neg)

                f_pos = np.sum(false_pos) / self.batch_size
                ep_false_pos.append(f_pos.item())
                f_neg = np.sum(false_neg) / self.batch_size
                ep_false_neg.append(f_neg.item())

                loss.backward()
                self.optimizer.step()
            print('Baigta epocha ', i, 'epochos nuostoliu suma %.3f ' % ep_loss,
                  'tikslumo vidurkis %.3f ' % np.mean(ep_acc), 'klaidingai teigiamu vidurkis %.3f ' %
                  np.mean(ep_false_pos), 'klaidingai neigiamu vidurkis %.3f ' % np.mean(ep_false_neg))

    """
        testavimo funkcija    
    """
    def test_cnn(self, verbose=True):
        if self.data_set_type == 'none':
            print('Nuotraukos nebuvo ikeltos (pasirinkite kitoki tinklo tipa)')
            return
        if verbose is False:
            print("Testuojamas tradicinio tipo tinklas. . .")
        self.train(mode=False)
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            ep_false_pos = []
            ep_false_neg = []
            for j, (pics, labels) in enumerate(self.data_loader):
                labels = labels.to(self.device)
                pics = pics.to(self.device)
                prediction = self.forward_pass(pics)
                prediction = prediction.to(self.device)
                loss = self.loss(prediction, labels)
                prediction = F.softmax(prediction, dim=0)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != labels,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_acc.append(acc.item())
                ep_loss += loss.item()

                false_pos = []
                false_neg = []
                for k in range(0, len(classes)):
                    if labels[k] == 0 and classes[k] == 1:
                        false_pos.append(1.)
                    else:
                        false_pos.append(0.)
                    if labels[k] == 1 and classes[k] == 0:
                        false_neg.append(1.)
                    else:
                        false_neg.append(0.)
                false_pos = np.array(false_pos)
                false_neg = np.array(false_neg)

                f_pos = np.sum(false_pos) / self.batch_size
                ep_false_pos.append(f_pos.item())
                f_neg = np.sum(false_neg) / self.batch_size
                ep_false_neg.append(f_neg.item())
            if verbose is True:
                print('Baigta epocha ', i, 'epochos nuostoliu suma %.3f ' % ep_loss,
                      'tikslumo vidurkis %.3f ' % np.mean(ep_acc), 'klaidingai teigiamu vidurkis %.3f ' %
                      np.mean(ep_false_pos), 'klaidingai neigiamu vidurkis %.3f ' % np.mean(ep_false_neg))
            self.loss_history.append(ep_loss)
            self.acc_history.append(np.mean(ep_acc))
            self.false_pos_history.append(np.mean(ep_false_pos))
            self.false_neg_history.append(np.mean(ep_false_neg))

        tools.save_accuracy_params(self, 'or_cnn', np.mean(self.acc_history), np.mean(self.false_pos_history),
                                   np.mean(self.false_neg_history))


if __name__ == "__main__":
    # inicializuojam modeli su treniravimo tipo duomenimis
    cnn = CNN(0.001, 50, 32, 'train', 40)
    # istreniruojam modeli - gausim svorius
    cnn.train_cnn()
    # issaugom svorius i diska
    tools.save_model(cnn, 'or_cnn')
    # gaunam issaugoto tinklo kelia diske
    path = tools.get_model_path_in_hdd(cnn, 'or_cnn')
    # is naujo inicializuojam modeli, tik jau su testiniais duomenim
    cnn = tools.load_model(path, 'or_cnn', 'test', 10)
    # testuojam modeli su testiniais duomenim, tikslumas bus issaugotas
    cnn.test_cnn()


