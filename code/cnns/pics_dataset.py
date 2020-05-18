import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils import data


class LungsDataSet(object):
    def __init__(self, data_set_type, batch_size):   # train, test, untouched
        pics_path = os.path.join(Path(Path(os.getcwd()).parent).parent, 'data', 'pics')
        images = []
        labels = []
        for root, dirs, files in os.walk(pics_path):
            self.num = 0
            for file in files:

                if data_set_type == 'train':
                    if 'subset_0' in file or 'subset_1' in file or 'subset_2' in file or 'subset_3' in file or \
                            'subset_4' in file or 'subset_5' in file or 'subset_6' in file or 'subset_7' in file:

                        img = Image.open(os.path.abspath(os.path.join(pics_path, file)))
                        img = np.array(img)
                        if img.size == 65 * 65:
                            images.append(img)
                            label = file[len(file) - 6]
                            labels.append(int(label))
                    self.num += 1

                if data_set_type == 'test':
                    if 'subset_8' in file:

                        img = Image.open(os.path.abspath(os.path.join(pics_path, file)))
                        img = np.array(img)
                        if img.size == 65 * 65:
                            images.append(img)
                            label = file[len(file) - 6]
                            labels.append(int(label))
                    self.num += 1

                if data_set_type == 'untouched':
                    if 'subset_9' in file:
                        img = Image.open(os.path.abspath(os.path.join(pics_path, file)))
                        img = np.array(img)
                        if img.size == 65 * 65:
                            images.append(img)
                            label = file[len(file) - 6]
                            labels.append(int(label))
                    self.num += 1
            images = images
            images = torch.tensor(images)
            labels = torch.tensor(labels)
        self.tensor_data_set = data.TensorDataset(images, labels)
        self.data_loader = data.DataLoader(self.tensor_data_set, batch_size=batch_size, shuffle=True, num_workers=8)

    def get_data(self):
        return self.tensor_data_set

    # unpack labels:
    # example:
    # 1, 0, 1, 1 -> 0 1, 1 0, 0 1, 0 1


"""
l = LungsDataSet('untouched')
print(type(l.get_data_loader().dataset))
print(l.get_data_loader().batch_size)
print(len(l.get_data_loader()))
"""
