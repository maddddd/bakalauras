import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
import numpy as np
import pics_dataset
import tools

"""
    VGG16 stiliaus tinklas. Originalus VGG16 tinklas analizuoja dideles nuotraukas (224x224 pikseliu). Sis modelis
    yra skirtas analizuoti mazesnems nuotraukoms (64x64 pikseliu), todel dalies sutraukimo (pooling) sluoksniu yra
    atsisakyta. Taip pat paskutiniai sluoksniai yra pakeisti isplestinemis konvoliucijomis.
    
    Hyperstulpelio ideja nera visai tinkama nuotrauku klasifikavimui, t.y. hyperstulpelis dazniau yra naudojamas
    nuotrauku segmentavimui (kiekvienas pikselis turi savo hyperstulpeli, kuris analizuojamas ir nustatoma jo verte,
    pvz. tai tinka plauciu konturui nustatyti. Taciau ideja pasinaudoti gilesniu sluoksniu duomenimis nera bloga.
    Galima konsensuso ideja - skirtingu konvoliucijos sluoksniu duomenis analizuojantys perceptronai turi prieiti kon-
    sensusa, kad nuotrauka butu pripazinta priklausancia vienai is dvieju klasiu.    
"""


class VGG16_CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, data_set_type, image_size=64):
        super(VGG16_CNN, self).__init__()
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
        self.acc_history_hyper = []
        self.false_pos_history_hyper = []
        self.false_neg_history_hyper = []
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device('cpu')

        """
            1) Tinklas is pradziu istreniruojamas kaip tradicinas VGG16 tipo tinklas, t.y. visi tinklo elementai yra
            treniruojami. Tam naudojamos visos konvoliucijos ir galu gale fc1_original perceptronas.
            2) Veliau tinklas treniruojamas imant pozymiu zemelapiu reiksmes skirtinguose tinklo gyliuose 
            (hyperstulpelis) ir treniruojamas su perceptronais fc1_hyper : fc5_hyper. 
            Siuo atveju konvoliuciniu filtru svoriai nebera atnaujinami, jie yra fiksuoti.
            
            Is pradziu uzregistruojam "hyper" atvejo perceptronus ir 2 atvejo optimizatoriu pries inicializuojant 
            kitus tinklo sluoksnius - tokiu budu 2 treniravimo atveju bus treniruojami tik "hyper" perceptronai. 
            Kiti tinklo elementai dar nera inicializuoti ir nebus pateikti self.parameters() f-jos, kuria naudojam 
            uzregistruojant tinklo elementus 2 tipo treniravimo optimizatoriui. 
            Alternatyva - visu nenorimu treniruoti tinklo elementu gradientu uzrakinimas (taip daroma uzrakinant jau
            istreniruotus "hyper" perceptronus, taciau patogiau, kai nereikia rakinti visu tinklo elementu).
        """
        self.input_dims_hyper = 1024    # fiksuotas dydis. Is anksto zinome, kad
        self.fc1_hyper = nn.Linear(self.input_dims_hyper, self.num_classes)
        self.fc2_hyper = nn.Linear(self.input_dims_hyper, self.num_classes)
        self.fc3_hyper = nn.Linear(self.input_dims_hyper, self.num_classes)
        self.fc4_hyper = nn.Linear(self.input_dims_hyper, self.num_classes)
        self.fc5_hyper = nn.Linear(self.input_dims_hyper, self.num_classes)
        self.fc6_hyper = nn.Linear(self.input_dims_hyper, self.num_classes)
        self.optimizer_hyper = optim.Adam(self.parameters(), lr=self.lr)

        self.conv1 = nn.Conv2d(1, 4, 3)  # grayscale, 8 konv. filtrai, 3x3 dydis
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
        # isplestos konvoliucijos
        self.dil_conv1 = nn.Conv2d(4, 8, 3, dilation=3)
        self.bn11 = nn.BatchNorm2d(8)
        self.dil_conv2 = nn.Conv2d(8, 4, 3, dilation=2)
        self.bn12 = nn.BatchNorm2d(4)

        self.input_dims_original = self.calc_input_dims_original()

        """
            Uzregistruojam tradicini optimizatoriu (1 treniravimo atvejis). Siuo atveju visi tinklo sluoksniai bus
            treniruojami, t.y. skaiciuojami ju gradientai.
        """
        self.fc1_original = nn.Linear(self.input_dims_original, self.num_classes)
        self.optimizer_original = optim.Adam(self.parameters(), lr=self.lr)

        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.data_set = None
        self.data_loader = None
        if self.data_set_type != 'none':
            self.data_set = pics_dataset.LungsDataSet(data_set_type, self.batch_size, 64)
            self.data_loader = self.data_set.data_loader

    def calc_input_dims_original(self):
        batch_data = T.zeros((1, 1, self.image_size, self.image_size))

        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.conv4(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)

        batch_data = self.conv7(batch_data)
        batch_data = self.maxpool2(batch_data)
        batch_data = self.conv8(batch_data)

        batch_data = self.conv9(batch_data)
        batch_data = self.conv10(batch_data)
        batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)

        batch_data = self.dil_conv1(batch_data)
        batch_data = self.dil_conv2(batch_data)

        return int(np.prod(batch_data.size()))

    def forward_pass_original(self, batch_data):
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

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv7(batch_data)
        batch_data = self.bn7(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.maxpool2(batch_data)

        batch_data = self.conv8(batch_data)
        batch_data = self.bn8(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv9(batch_data)
        batch_data = self.bn9(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv10(batch_data)
        batch_data = self.bn10(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)

        batch_data = self.dil_conv1(batch_data)
        batch_data = self.bn11(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.dil_conv2(batch_data)
        batch_data = self.bn12(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)

        classes = self.fc1_original(batch_data)

        return classes

    def forward_pass_hyper(self, batch_data, hypercolumn_index):
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

        if hypercolumn_index == 0:
            resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)
            hypercolumn = resized_batch_data[:, :, 24:40, 24:40].flatten()
            hypercolumn = hypercolumn.view(self.batch_size, -1)
            classes = self.fc1_hyper(hypercolumn)
            return classes

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        if hypercolumn_index == 1:
            resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)
            hypercolumn = resized_batch_data[:, :, 24:40, 24:40].flatten()
            hypercolumn = hypercolumn.view(self.batch_size, -1)
            classes = self.fc2_hyper(hypercolumn)
            return classes

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        if hypercolumn_index == 2:
            resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)
            hypercolumn = resized_batch_data[:, :, 24:40, 24:40].flatten()
            hypercolumn = hypercolumn.view(self.batch_size, -1)
            classes = self.fc3_hyper(hypercolumn)
            return classes

        batch_data = self.conv7(batch_data)
        batch_data = self.bn7(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.maxpool2(batch_data)

        batch_data = self.conv8(batch_data)
        batch_data = self.bn8(batch_data)
        batch_data = F.relu(batch_data)

        if hypercolumn_index == 3:
            resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)
            hypercolumn = resized_batch_data[:, :, 24:40, 24:40].flatten()
            hypercolumn = hypercolumn.view(self.batch_size, -1)
            classes = self.fc4_hyper(hypercolumn)
            return classes

        batch_data = self.conv9(batch_data)
        batch_data = self.bn9(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv10(batch_data)
        batch_data = self.bn10(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)

        if hypercolumn_index == 4:
            hypercolumn = batch_data[:, :, 24:40, 24:40].flatten()
            hypercolumn = hypercolumn.view(self.batch_size, -1)
            classes = self.fc5_hyper(hypercolumn)
            return classes

        batch_data = self.dil_conv1(batch_data)
        batch_data = self.bn11(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.dil_conv2(batch_data)
        batch_data = self.bn12(batch_data)
        batch_data = F.relu(batch_data)

        if hypercolumn_index == 5:
            resized_batch_data = F.interpolate(batch_data, size=(64, 64), mode='bilinear', align_corners=True)
            hypercolumn = resized_batch_data[:, :, 24:40, 24:40].flatten()
            hypercolumn = hypercolumn.view(self.batch_size, -1)
            classes = self.fc6_hyper(hypercolumn)
            return classes

    def train_cnn(self):
        if self.data_set_type == 'none':
            print('Nuotraukos nebuvo ikeltos (pasirinkite kitoki tinklo tipa)')
            return
        self.train(mode=True)

        # 1 treniravimo zingsnis:
        print("1 TRENIRAVIMO ZINGSNIS: treniruojamas tradicinio VGG16 stiliaus tinklas\n")
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            ep_false_pos = []
            ep_false_neg = []
            for j, (pics, labels) in enumerate(self.data_loader):
                self.optimizer_original.zero_grad()
                labels = labels.to(self.device)
                pics = pics.to(self.device)
                if pics.size()[0] == self.batch_size:
                    prediction = self.forward_pass_original(pics)
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
                    self.optimizer_original.step()
            print('Baigta epocha ', i, 'epochos nuostoliu suma %.3f ' % ep_loss,
                  'tikslumo vidurkis %.3f ' % np.mean(ep_acc), 'klaidingai teigiamu vidurkis %.3f ' %
                  np.mean(ep_false_pos), 'klaidingai neigiamu vidurkis %.3f ' % np.mean(ep_false_neg))

        # 2 treniravimo zingsnis:
        print("\n2 TRENIRAVIMO ZINGSNIS: treniruojami perceptronai su skirtingu gyliu duomenimis")
        for h in range(0, 6):
            print("\nTreniruojamas %d lygio hyperkolonos perceptronas\n" % h)
            self.lock_fc_layers(h)
            for i in range(self.epochs):
                ep_loss = 0
                ep_acc = []
                ep_false_pos = []
                ep_false_neg = []
                for j, (pics, labels) in enumerate(self.data_loader):
                    self.optimizer_hyper.zero_grad()
                    labels = labels.to(self.device)
                    pics = pics.to(self.device)
                    if pics.size()[0] == self.batch_size:
                        prediction = self.forward_pass_hyper(pics, h)
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
                        self.optimizer_hyper.step()
                print('Baigta epocha ', i, 'epochos nuostoliu suma %.3f ' % ep_loss,
                      'tikslumo vidurkis %.3f ' % np.mean(ep_acc), 'klaidingai teigiamu vidurkis %.3f ' %
                      np.mean(ep_false_pos), 'klaidingai neigiamu vidurkis %.3f ' % np.mean(ep_false_neg))

    def test_cnn(self, verbose=True, hyper_test_iters=5):
        if self.data_set_type == 'none':
            print('Nuotraukos nebuvo ikeltos (pasirinkite kitoki tinklo tipa)')
            return
        if verbose is False:
            print("Testuojamas VGG16 tipo tinklas. . .")
        self.train(mode=False)
        if verbose is True:
            print("\n1 TESTAVIMO ETAPAS: nustatomi testavimo hyperkolonos sluoksniuose rezultatai")
        for h in range(0, 6):
            if verbose is True:
                print("\nTestuojamas %d hyperkolonos sluoksnis\n" % h)
            hyper_acc = []
            hyper_false_pos = []
            hyper_false_neg = []
            for i in range(hyper_test_iters):
                ep_loss = 0
                ep_acc = []
                ep_false_pos = []
                ep_false_neg = []
                for j, (pics, labels) in enumerate(self.data_loader):
                    labels = labels.to(self.device)
                    pics = pics.to(self.device)
                    if pics.size()[0] == self.batch_size:
                        prediction = self.forward_pass_hyper(pics, h)
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
                hyper_acc.append(np.mean(ep_acc))
                hyper_false_pos.append(np.mean(ep_false_pos))
                hyper_false_neg.append(np.mean(ep_false_neg))
            self.acc_history_hyper.append(np.mean(hyper_acc))
            self.false_pos_history_hyper.append(np.mean(hyper_false_pos))
            self.false_neg_history_hyper.append(np.mean(hyper_false_neg))

        if verbose is True:
            print("\n1 TESTAVIMO etapas baigtas. Gauti tokie tinklo sluoksniu tikslumai:")
        for i in range(6):
            if verbose is True:
                print("%d sluoksnis: " % i, "tikslumas %.3f" % self.acc_history_hyper[i],
                      "klaidingai teigiamu vidurkis %.3f" % self.false_pos_history_hyper[i],
                      "klaidingai neigiamu vidurkis %.3f" % self.false_neg_history_hyper[i])

        if verbose is True:
            print("\n2 TESTAVIMO ETAPAS: klasifikuojama testavimo duomenu aibe, atsizvelgiant i "
                  "hyperkolonu konsensusa\n")

        for i in range(self.epochs):
            ep_acc = []
            ep_false_pos = []
            ep_false_neg = []
            for j, (pics, labels) in enumerate(self.data_loader):
                labels = labels.to(self.device)
                pics = pics.to(self.device)
                if pics.size()[0] == self.batch_size:
                    all_predictions = T.zeros((self.batch_size, 2))
                    for h in range(6):
                        prediction = self.forward_pass_hyper(pics, h)
                        prediction = prediction.to(self.device)
                        prediction = F.softmax(prediction, dim=0)
                        all_predictions += prediction * self.acc_history_hyper[h]

                    classes = T.argmax(all_predictions, dim=1)   # grazina didziausio argumento indeksa, o ne reiksme

                    wrong = T.where(classes != labels,
                                    T.tensor([1.]).to(self.device),
                                    T.tensor([0.]).to(self.device))
                    acc = 1 - T.sum(wrong) / self.batch_size

                    ep_acc.append(acc.item())

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
                print('Baigta epocha ', i, 'tikslumo vidurkis %.3f ' % np.mean(ep_acc),
                      'klaidingai teigiamu vidurkis %.3f ' % np.mean(ep_false_pos),
                      'klaidingai neigiamu vidurkis %.3f ' % np.mean(ep_false_neg))
            self.acc_history.append(np.mean(ep_acc))
            self.false_pos_history.append(np.mean(ep_false_pos))
            self.false_neg_history.append(np.mean(ep_false_neg))

        tools.save_accuracy_params(self, 'vgg16_cnn', np.mean(self.acc_history), np.mean(self.false_pos_history),
                                   np.mean(self.false_neg_history))

    """
        Gradientu uzrakinimas visiems hyperkolona treniruojantiems perceptronams, isskyrus nurodytajam.
        Tai reiskia, kad visi perceptronai, isskyrus nurodytaji, nebus treniruojami, t.y. ju svoriai liks uzrakinti.
        Reikalinga tam, kad tolimesnio sluoksnio perceptrono treniravimas nesugadintu anktesniu sluoksniu duomenimis
        istreniruotu perceptronu, t.y. jie jau yra istreniruoti, pertreniruoti ju nebereikia.
    """

    def lock_fc_layers(self, keep_unlocked):
        all_fcs = [self.fc1_hyper, self.fc2_hyper, self.fc3_hyper, self.fc4_hyper, self.fc5_hyper, self.fc6_hyper]
        for i in range(0, 6):
            if i != keep_unlocked:
                for param in all_fcs[i].parameters():
                    param.requires_grad = False
            else:
                for param in all_fcs[i].parameters():
                    param.requires_grad = True


if __name__ == "__main__":
    # inicializuojam modeli su treniravimo tipo duomenimis
    cnn = VGG16_CNN(0.001, 100, 32, 'train', 64)
    # istreniruojam modeli - gausim svorius
    cnn.train_cnn()
    # issaugom svorius i diska
    tools.save_model(cnn, 'vgg16_cnn')
    # gaunam issaugoto tinklo kelia diske
    path = tools.get_model_path_in_hdd(cnn, 'vgg16_cnn')
    # is naujo inicializuojam modeli, tik jau su testiniais duomenim
    cnn = tools.load_model(path, 'vgg16_cnn', 'test', 100)
    # testuojam modeli su testiniais duomenim, tikslumas bus issaugotas
    cnn.test_cnn()
