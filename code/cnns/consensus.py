import os
from pathlib import Path
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import or_cnn
import mgi_cnn
import vgg16_cnn
import tools
import pics_dataset


def resize_tensor(cnn, tensor):
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
        temp = 0
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            ep_false_pos = []
            ep_false_neg = []
            for j, (pics, labels) in enumerate(self.data_loader):
                print(pics.size())
                if self.mode == 'majority':
                    for j in range(len(self.cnns)):
                        pics = resize_tensor(self.cnns[j], pics)
                        prediction = self.cnns[j].forward_pass(pics)
                        prediction = F.softmax(prediction, dim=0)
                        classes = T.argmax(prediction, dim=1)
                    temp += 1
                if self.mode == 'weighted':
                    print('todo')


if __name__ == "__main__":
    nets, accs, pos, negs = load_networks_from_paths()

    cons_net = Consensus(nets, accs, pos, negs, 'majority', 0.001, 1, 32)
    cons_net.predict()
