#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2022/8/6 23:48
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : make_random_data.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080
import pickle
import random

from utils.util import setup_seed

if __name__ == "__main__":
    for seed in [11, 33]:
        setup_seed(seed)
        for task in ["1_02"]:
            for mode in ["val", "real_test", "train"]:
                with open(
                    "../../isic_data/sxf_{}_data_{}_normal_224.pkl".format(mode, task),
                    "rb",
                ) as f:
                    data = pickle.load(f)

                img_data = []
                labels = []
                img_data1 = []
                labels1 = []
                img_data2 = []
                labels2 = []

                for k in data.keys():
                    for img in data[k]:
                        if k == "1":
                            img_data1.append(img)
                            labels1.append(1)
                        if k == "02":
                            img_data2.append(img)
                            labels2.append(0)

                random.shuffle(img_data1)
                random.shuffle(img_data2)

                data_1_02 = {"1": img_data1, "02": img_data2}

                with open(
                    "../../isic_data/sxf_{}_data_{}_normal_224_s{}.pkl".format(
                        mode, task, seed
                    ),
                    "wb",
                ) as f:
                    pickle.dump(data_1_02, f)

    for seed in [11, 33]:
        setup_seed(seed)
        for task in ["2_01"]:
            for mode in ["val", "real_test", "train"]:
                with open(
                    "../../isic_data/sxf_{}_data_{}_normal_224.pkl".format(mode, task),
                    "rb",
                ) as f:
                    data = pickle.load(f)

                img_data = []
                labels = []
                img_data1 = []
                labels1 = []
                img_data2 = []
                labels2 = []

                for k in data.keys():
                    for img in data[k]:
                        if k == "2":
                            img_data1.append(img)
                            labels1.append(1)
                        if k == "01":
                            img_data2.append(img)
                            labels2.append(0)

                random.shuffle(img_data1)
                random.shuffle(img_data2)

                data_1_02 = {"2": img_data1, "01": img_data2}

                with open(
                    "../../isic_data/sxf_{}_data_{}_normal_224_s{}.pkl".format(
                        mode, task, seed
                    ),
                    "wb",
                ) as f:
                    pickle.dump(data_1_02, f)
