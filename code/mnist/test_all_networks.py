import os
from pathlib import Path


def test_networks():
    find_network('mask_r_cnn')
    find_network('mgi_cnn')
    find_network('original_cnn')
    find_network('vgg16_cnn')


def find_network(name):
    net = find_trained_net(name)
    if net is None:
        net = train_network(name)
    test_network(net)


def find_trained_net(name):
    code_folder_path = Path(os.getcwd()).parent
    trained_net_path = os.path.abspath(os.path.join(code_folder_path, 'mnist'))
    trained_net_path = os.path.abspath(os.path.join(trained_net_path, 'trained_nets'))
    trained_net_path = os.path.abspath(os.path.join(trained_net_path, name + '.yaml'))
    if os.path.exists(trained_net_path):
        return 'test'
    else:
        return None


def train_network(name):
    print('training network: ' + name)
    return 'result'


def test_network(net):
    print('testing network: ' + net)