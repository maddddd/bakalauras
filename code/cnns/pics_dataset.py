import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils import data


# from numpy array
def cut_image(image, new_size):
    if new_size == 65:
        return image
    start_index = int((len(image) - new_size) / 2)
    end_index = start_index + new_size
    return image[start_index:end_index, start_index:end_index]


class LungsDataSet(object):
    def __init__(self, data_set_type, batch_size, image_size):   # train, test, untouched
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
                        if image_size != 65:
                            img = cut_image(img, image_size)
                        if img.size == image_size * image_size:
                            images.append(img)
                            label = file[len(file) - 6]
                            labels.append(int(label))
                    self.num += 1

                if data_set_type == 'test':
                    if 'subset_8' in file:

                        img = Image.open(os.path.abspath(os.path.join(pics_path, file)))
                        img = np.array(img)
                        if image_size != 65:
                            img = cut_image(img, image_size)
                        if img.size == image_size * image_size:
                            images.append(img)
                            label = file[len(file) - 6]
                            labels.append(int(label))
                    self.num += 1

                if data_set_type == 'untouched':
                    if 'subset_9' in file:
                        img = Image.open(os.path.abspath(os.path.join(pics_path, file)))
                        img = np.array(img)
                        if image_size != 65:
                            img = cut_image(img, image_size)
                        if img.size == image_size * image_size:
                            images.append(img)
                            label = file[len(file) - 6]
                            labels.append(int(label))
                    self.num += 1
            images = torch.tensor(images)
            labels = torch.tensor(labels)
        self.tensor_data_set = data.TensorDataset(images, labels)
        self.data_loader = data.DataLoader(self.tensor_data_set, batch_size=batch_size, shuffle=True, num_workers=8)

