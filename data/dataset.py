#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2022/8/3 20:38
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : dataset.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_set, labels, person, nomal):
        self.data_set = np.array(data_set)
        self.labels = labels
        self.nomal = nomal
        self.person = person

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        if self.nomal:
            # -0.6是为了把数据变成-1到1(-0.6,0.6)之间
            return np.array([self.data_set[idx] - 0.6]), self.labels[idx]
        return np.array(self.data_set[idx]), self.labels[idx], self.person[idx]
