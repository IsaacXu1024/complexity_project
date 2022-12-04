import numpy as np
import torch
import PIL

import torchvision.datasets as datasets
from torch.utils.data import Dataset

from utils.utils import list_all_equal

class FilteredCIFAR10Dataset(Dataset):
    _BASE_CLASS_COUNT = 2

    _CIFAR10_CLASS_DICT = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }

    def __init__(self, classes=None, transform=None, train=True):
        self.train = train
        self.cifar10 = datasets.CIFAR10(
            root='../../data',
            download=True,
            train=train,
            transform=transform
        )
        self.data = self.cifar10.data
        self.tgts = self.cifar10.targets
        self.f_data, self.f_tgts = self.data, self.tgts
        self.transform = transform
        if classes:
            self.classes = classes
            self.class_limits = self.find_balance()
            remove_list = self.gen_remove_list()
            self.f_data, self.f_tgts = self.__remove__(remove_list)
        #self.sanity_check()

    def __getitem__(self, idx):
        data, tgt = self.f_data[idx], self.f_tgts[idx]
        return idx, self.transform(PIL.Image.fromarray(data)), tgt

    def __len__(self):
        return len(self.f_data)
    
    def __remove__(self, remove_list):
        mask = np.ones(len(self.data), dtype=bool)
        mask[remove_list] = False
        return self.data[mask], self.update_labs(np.asarray(self.tgts)[mask])
    
    def gen_remove_list(self):
        remove_list = []
        class_counter = {k: 0 for k in self.class_limits.keys()}

        for i, tgt in enumerate(self.tgts):
            if tgt not in class_counter or class_counter[tgt] >= self.class_limits[tgt]:
                remove_list.append(i)
            else: class_counter[tgt] += 1 
        return remove_list

    def update_labs(self, tgts):
        inv_class_dict = {v: k for k, v in self._CIFAR10_CLASS_DICT.items()}
        updated_tgts = []
        for tgt in tgts:
            updated_tgts.append(self.classes[inv_class_dict[tgt]])
        return updated_tgts

    def find_balance(self):
        if self.train: class_len = 5000
        else: class_len = 1000

        class_count = np.asarray([sum(1 for c in self.classes.values() if c == i) for i in range(len(self.classes))])
        if list_all_equal(class_count):
            class_limit_dict = {self._CIFAR10_CLASS_DICT[k]: class_len*self._BASE_CLASS_COUNT/len(self.classes) for k in self.classes.keys()}
        else:
            class_count = [int(round(class_len/count)) if count != 0 else 0 for count in class_count]
            class_limit_dict = {self._CIFAR10_CLASS_DICT[k]: class_count[v] for k, v in self.classes.items()}
        return class_limit_dict

    def sanity_check(self):
        assert len(self.f_data) == len(self.f_tgts)
        counter = {}
        for tgt in self.f_tgts:
            if tgt in counter:
                counter[tgt] += 1
            else: counter[tgt] = 1
        print(counter)