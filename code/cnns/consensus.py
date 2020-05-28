import os
from pathlib import Path
import torch as T
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import mgi_cnn
import or_cnn
import tools
import pics_dataset


def resize_tensor(cnn, tensor):
    """
        nuotraukos tenzoriaus apkarpymas. Treniruojant tinklus, jie galejo tureti skirtingus pradinius nuotrauku
        dydzius, todel butina pritaikyti ivesti.
    """
    start = int((64 - cnn.image_size) / 2)
    end = int(start + cnn.image_size)
    tensor = tensor[:, start:end, start:end]
    return tensor


def get_all_saved_net_paths():
    """
        gauname visu tinklu, kurie yra issaugoti, absoliucius kelius. Ne visi tinklai tinkami naudoti konsensuse - jie
        is pradziu turi buti istestuoti (o testavimo tikslumai irasyti faile params.txt)
    """
    all_nets = []
    path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets'))
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pt'):
                all_nets.append(file)
    return all_nets


def get_all_tested_net_paths(net_paths):
    """
        gauname visu tinklu, kurie yra istestuoti, absoliucius kelius. Padavus visu issaugotu tinklu keliu list'a kaip
        parametra, gausime visus tinklus, kurie yra tinkami konsensuso testavimui (t.y. turi tiek svorius, tiek ir
        testavimo rezultatus).
    """
    tested_net_paths = []
    path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets', 'params.txt'))
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            net_in_file = line.split(' ')[0] + '.pt'
            for net_path in net_paths:
                if net_path == net_in_file:
                    tested_net_paths.append(net_path)
                    break
    return tested_net_paths


def get_nets_by_indices():
    nets = get_all_tested_net_paths(get_all_saved_net_paths())
    print('Galimi tinklai:\n')
    for net in nets:
        print(net)
    print('Iveskite norimu tinklu indeksus:\n')
    """
        Pvz. '1 2 3'
    """
    chosen_nets = []
    input_indices = input()
    print('\n')
    indices = input_indices.split(' ')
    for index in indices:
        try:
            if 0 <= int(index) < len(nets):
                chosen_nets.append(nets[int(index)])
        except:
            print('Indeksas privalo buti sveikas skaicius')

    print('Pasirinkti tinklai:\n')
    for net in chosen_nets:
        print(net)
    print('\n')
    return chosen_nets


def load_networks_from_paths():
    paths = get_nets_by_indices()
    loaded_nets = []
    accuracies = []
    false_pos = []
    false_negs = []
    vgg_accuracies = []
    path_in_disk = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets'))
    params_file_path = path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets',
                                                           'params.txt'))
    for p in paths:
        # surandam modeli:
        if 'or_cnn' in p:
            path = os.path.abspath(os.path.join(path_in_disk, p))
            cnn = tools.load_model(path, 'or_cnn', 'none', 0)
            loaded_nets.append(cnn)
        if 'mgi_cnn' in p:
            path = os.path.abspath(os.path.join(path_in_disk, p))
            cnn = tools.load_model(path, 'mgi_cnn', 'none', 0)
            loaded_nets.append(cnn)
        if 'vgg16_cnn' in p:
            path = os.path.abspath(os.path.join(path_in_disk, p))
            cnn = tools.load_model(path, 'vgg16_cnn', 'none', 0)
            loaded_nets.append(cnn)
        # surandam jo tikslumo duomenis:
        p = p[0:len(p)-3]
        with open(params_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                if line[0] == p:
                    accuracies.append(float(line[1]))
                    false_pos.append(float(line[2]))
                    false_negs.append(float(line[3]))
                    if 'vgg' in p:
                        loaded_nets[len(loaded_nets)-1].acc_history_hyper.append(float(line[4]))
                        loaded_nets[len(loaded_nets) - 1].acc_history_hyper.append(float(line[5]))
                        loaded_nets[len(loaded_nets) - 1].acc_history_hyper.append(float(line[6]))
                        loaded_nets[len(loaded_nets) - 1].acc_history_hyper.append(float(line[7]))
                        loaded_nets[len(loaded_nets) - 1].acc_history_hyper.append(float(line[8]))
                        loaded_nets[len(loaded_nets) - 1].acc_history_hyper.append(float(line[9]))
    return loaded_nets, accuracies, false_pos, false_negs


class Consensus:
    def __init__(self, cnns, accs, pos, negs, mode, lr, epochs, batch_size, data_set_type='untouched'):
        self.cnns = cnns
        self.data_set = pics_dataset.LungsDataSet(data_set_type, batch_size, 64)
        self.data_loader = self.data_set.data_loader
        self.mode = mode
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = 2
        self.loss_history = []
        self.acc_history = []
        self.false_pos_history = []
        self.false_neg_history = []
        self.loss = nn.CrossEntropyLoss()

    def predict(self):
        """
            Nuotrauku klasifikavimo metodas.

            mode tipai:
            'majority'          - grazina klases, t.y. 1 arba 0. Naudojama daugumos balsavimo atveju.
            'weighted'          - grazina softmax funkcijos reiksmes. Jos sudauginamos su tikslumo istorijos duomenimis
                                ir suformuojamas galutinis klasifikavimo rezultatas.
        """
        if self.mode == 'majority' and len(self.cnns) % 2 == 0:
            print('Norint naudoti daugumos balsavima, tinklu skaicius privalo buti nelyginis')
            return
        for i in range(self.epochs):
            ep_acc = []
            ep_false_pos = []
            ep_false_neg = []
            for j, (pics, labels) in enumerate(self.data_loader):
                if pics.size()[0] == self.batch_size:
                    if self.mode == 'majority':
                        all_predictions = T.zeros(self.batch_size)
                        for j in range(len(self.cnns)):
                            resized_pics = resize_tensor(self.cnns[j], pics)
                            if isinstance(self.cnns[j], mgi_cnn.MGI_CNN):
                                pics_20x20, pics_30x30, pics_40x40 = mgi_cnn.get_resized_images(resized_pics)
                                prediction = self.cnns[j].forward_pass(pics_20x20, pics_30x30, pics_40x40)
                            else:
                                if isinstance(self.cnns[j], or_cnn.CNN):
                                    prediction = self.cnns[j].forward_pass(resized_pics)
                                else:
                                    prediction = T.zeros((self.batch_size, 2))
                                    for h in range(6):
                                        hyper_prediction = self.cnns[j].forward_pass_hyper(pics, h)
                                        hyper_prediction = F.softmax(prediction, dim=0)
                                        prediction += hyper_prediction * self.cnns[j].acc_history_hyper[h]

                            prediction = F.softmax(prediction, dim=0)
                            classes = T.argmax(prediction, dim=1)
                            all_predictions += classes
                        wrong = 0
                        false_pos = 0
                        false_neg = 0
                        consensus_barrier = len(self.cnns) / 2
                        for k in range(len(all_predictions)):
                            if labels[k] == 1:
                                if all_predictions[k] < consensus_barrier:
                                    wrong += 1
                                    false_neg += 1
                            else:
                                if all_predictions[k] > consensus_barrier:
                                    wrong += 1
                                    false_pos += 1
                        acc = 1 - wrong / self.batch_size
                        ep_acc.append(acc)

                        false_pos = false_pos / self.batch_size
                        false_neg = false_neg / self.batch_size
                        ep_false_pos.append(false_pos)
                        ep_false_neg.append(false_neg)

                    if self.mode == 'weighted':
                        all_predictions = T.zeros(self.batch_size, 2)
                        for j in range(len(self.cnns)):
                            resized_pics = resize_tensor(self.cnns[j], pics)
                            if isinstance(self.cnns[j], mgi_cnn.MGI_CNN):
                                pics_20x20, pics_30x30, pics_40x40 = mgi_cnn.get_resized_images(resized_pics)
                                prediction = self.cnns[j].forward_pass(pics_20x20, pics_30x30, pics_40x40)
                            else:
                                if isinstance(self.cnns[j], or_cnn.CNN):
                                    prediction = self.cnns[j].forward_pass(resized_pics)
                                else:
                                    prediction = T.zeros((self.batch_size, 2))
                                    for h in range(6):
                                        hyper_prediction = self.cnns[j].forward_pass_hyper(pics, h)
                                        hyper_prediction = F.softmax(prediction, dim=0)
                                        prediction += hyper_prediction * self.cnns[j].acc_history_hyper[h]
                            prediction = F.softmax(prediction, dim=0)
                            all_predictions += prediction * accs[j]
                        wrong = 0
                        false_pos = 0
                        false_neg = 0
                        predicted_classes = T.zeros(self.batch_size)
                        for k in range(self.batch_size):
                            if labels[k] == 1:
                                if all_predictions[k, 0] > all_predictions[k, 1]:
                                    wrong += 1
                                    false_neg += 1
                            else:
                                if all_predictions[k, 1] > all_predictions[k, 0]:
                                    wrong += 1
                                    false_pos += 1
                        acc = 1 - wrong / self.batch_size
                        ep_acc.append(acc)

                        false_pos = false_pos / self.batch_size
                        false_neg = false_neg / self.batch_size
                        ep_false_pos.append(false_pos)
                        ep_false_neg.append(false_neg)

            print('Baigta epocha ', i, 'tikslumo vidurkis %.3f ' % np.mean(ep_acc),
                  'klaidingai teigiamu vidurkis %.3f ' % np.mean(ep_false_pos),
                  'klaidingai neigiamu vidurkis %.3f ' % np.mean(ep_false_neg))


if __name__ == "__main__":
    nets, accs, pos, negs = load_networks_from_paths()

    cons_net = Consensus(nets, accs, pos, negs, 'weighted', 0.001, 5, 32)
    cons_net.predict()
