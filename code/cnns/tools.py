import os
from pathlib import Path
import torch
import or_cnn
import mgi_cnn
import mask_r_cnn
import vgg16_cnn


def save_model(model, model_type):
    path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets'))
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.abspath(os.path.join(path, model_type + '.pt'))
    torch.save({'state_dict': model.state_dict()}, model_path)


def load_or_cnn_model(lr, epochs, batch_size, path):
    loaded_cnn = or_cnn.CNN(lr, epochs, batch_size, 'untouched')
    state_dict = torch.load(path)['state_dict']
    loaded_cnn.load_state_dict(state_dict)
    return loaded_cnn


def load_mgi_cnn_model():
    print("TODO")


def load_vgg16_cnn_model():
    print("TODO")


def load_mask_r_cnn_model():
    print("TODO")
