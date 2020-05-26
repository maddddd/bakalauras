import os
from pathlib import Path
import torch
import or_cnn
import mgi_cnn
import vgg16_cnn


def save_model(model, model_type):
    """
        modelio issaugojimas.
        model       - pats modelis (objektas)
        model_type  - 'or_cnn', 'mgi_cnn', 'vgg16_cnn'
    """
    path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets'))
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.abspath(os.path.join(path, model_type + '_lr_' + str(model.lr) + '_epochs_' + str(model.epochs)
                                              + '_batch_size_' + str(model.batch_size) + '_image_size_' +
                                              str(model.image_size) + '.pt'))
    torch.save({'state_dict': model.state_dict()}, model_path)


def load_model(path, model_type, dataset_type, new_epochs):
    """
        modelio inicializavimas is failo. Failo vardas turi buti sugeneruotas
        save_model f-jos, t.y. savavaliskai negalima jo keisti.
        path            - modelio failo absoliutus kelias diske
        model_type      - 'or_cnn', 'mgi_cnn', 'vgg16_cnn'
        dataset_type    - 'train', 'test', 'untouched'
        new_epochs - naujas epochu skaicius modelyje (testavimui, tolimesniam treniravimui ir t.t.).
    """
    params = path.split("_")
    lr, batch_size, image_size = 0.001, 48, 64
    for i in range(0, len(params)):
        if params[i] == 'lr':
            lr = float(params[i+1])
        else:
            if params[i] == 'batch':
                batch_size = int(params[i+2])
            else:
                if params[i] == 'image':
                    image_size = int(params[i+2].split('.')[0])
    # print(str(lr) + ' ' + str(batch_size) + ' ' + str(image_size))
    loaded_cnn = None
    if model_type == 'or_cnn':
        loaded_cnn = or_cnn.CNN(lr, new_epochs, batch_size, dataset_type, image_size)
    else:
        if model_type == 'mgi_cnn':
            loaded_cnn = mgi_cnn.MGI_CNN(lr, new_epochs, batch_size, dataset_type)
        else:
            if model_type == 'vgg16_cnn':
                loaded_cnn = vgg16_cnn.VGG16_CNN(lr, new_epochs, batch_size, dataset_type)
    state_dict = torch.load(path)['state_dict']
    loaded_cnn.load_state_dict(state_dict)
    return loaded_cnn


def save_accuracy_params(model, model_type, acc, false_pos, false_neg):
    path = os.path.abspath(os.path.join(Path(os.getcwd()).parent.parent, 'trained_nets'))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.abspath(os.path.join(path, 'params.txt'))

    # sukuriam params.txt faila, kuriame laikome tikslumo duomenis (jei jo dar nera)
    if not os.path.isfile(path):
        open(path, "w+").close()

    name = model_type + '_lr_' + str(model.lr) + '_epochs_' + str(model.epochs) + '_batch_size_' + \
           str(model.batch_size) + '_image_size_' + str(model.image_size)

    # patikrinam, ar tokie duomenys dar neegzistuoja
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            if line[0] == name:
                return

    # jei neegzistuoja, sukuriam:
    with open(path, 'a') as f:
        f.write(name + " %.5f " % acc + "%.5f " % false_pos + "%.5f " % false_neg + "\n")


def get_model_path_in_hdd(model):
    name = ''
    if model is mgi_cnn.MGI_CNN:
        name = 'mgi_cnn'
    if model is or_cnn.CNN:
        name = 'or_cnn'
    if model is vgg16_cnn.VGG16_CNN:
        name = 'vgg16_cnn'