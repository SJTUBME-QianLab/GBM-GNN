#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 21:21
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : data.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080

import pickle

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import random

from utils.util import calculate_w


class MyDataset(Dataset):
    def __init__(self, data_set, labels, person, normal):
        self.data_set = data_set
        self.labels = labels
        self.normal = normal
        self.person = person

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        if self.normal:
            # -0.6是为了把数据变成-1到1(-0.6,0.6)之间
            return np.array([self.data_set[idx] - 0.6]), self.labels[idx]
        return np.array([self.data_set[idx]]), self.labels[idx], self.person[idx]


class JCD_Dataset(Dataset):
    def __init__(self, dataset, label, normal, iteration):

        if normal:
            dataset = np.array(dataset) - 0.6
        else:
            dataset = np.array(dataset)
        self.data = dataset[:, np.newaxis, :, :]
        self.label = label
        self.iteration = iteration
        print(self.data.shape, np.array(self.label).shape)

        self.dis_data_score0 = []
        self.dis_data_score1 = []

        self.sk_data_score0 = []
        self.sk_data_score1 = []

        self.label_score0 = []
        self.label_score1 = []

        for i in range(len(self.label)):
            if self.label[i] == 0:
                self.sk_data_score0.append(self.data[i])
                self.label_score0.append(self.label[i])
            elif self.label[i] == 1:
                self.sk_data_score1.append(self.data[i])
                self.label_score1.append(self.label[i])

        self.label0_index_box = np.arange(len(self.label_score0))
        self.label1_index_box = np.arange(len(self.label_score1))

    # initial coding
    def __getitem__(self, index):
        # 按照序号index返回数据和标签
        # 无放回抽取一个0类
        if index % 2 == 0:
            # 获取一下盒子中剩余元素的个数
            length = len(self.label0_index_box)
            # 如果盒子为空，则重新填满
            if length == 0:
                self.label0_index_box = np.arange(len(self.label_score0))
                length = len(self.label0_index_box)
            # 从盒子中剩余的样本中随机抽取一个
            random_index = random.randint(0, length - 1)
            sk = self.sk_data_score0[self.label0_index_box[random_index]]
            y = self.label_score0[self.label0_index_box[random_index]]
            # 将抽出的样本序号从盒子中删除
            self.label0_index_box = np.delete(self.label0_index_box, random_index)
        elif index % 2 == 1:
            length = len(self.label1_index_box)
            if length == 0:
                self.label1_index_box = np.arange(len(self.label_score1))
                length = len(self.label1_index_box)
            random_index = random.randint(0, length - 1)
            sk = self.sk_data_score1[self.label1_index_box[random_index]]
            y = self.label_score1[self.label1_index_box[random_index]]
            self.label1_index_box = np.delete(self.label1_index_box, random_index)

        sk = torch.from_numpy(sk).float()
        return sk, y

    def __len__(self):
        # 返回数据库长度
        return self.iteration


class Train_Generator(DataLoader):
    def __init__(self, root, keys=["0", "1"]):

        with open(root, "rb") as load_data:
            data_dict = pickle.load(load_data)
        data_ = {}
        for i in range(len(keys)):
            data_[i] = data_dict[keys[i]]

        print(len(data_[0]), len(data_[1]))

        self.data = data_
        self.channel = 1
        self.feature_shape = np.array((self.data[1][0])).shape

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(
        self,
        batch_size=5,
        n_way=2,
        num_shots=10,
        unlabeled_extra=0,
        cuda=True,
        variable=False,
    ):
        # init # features
        batch_x = np.zeros(
            (batch_size, self.channel, self.feature_shape[1], self.feature_shape[2]),
            dtype="float32",
        )
        # labels
        labels_x = np.zeros((batch_size, n_way), dtype="float32")
        labels_x_global = np.zeros(batch_size, dtype="int64")
        num = n_way * num_shots + 1

        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros(
                    (
                        batch_size,
                        self.channel,
                        self.feature_shape[1],
                        self.feature_shape[2],
                    ),
                    dtype="float32",
                )
            )
            labels_yi.append(np.zeros((batch_size, n_way), dtype="float32"))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype="float32")))

        # feed data

        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if (
                    class_num == pre_class
                ):  # 如果和随机选择的类别class_counter和随机产生的类号 positive_class
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    batch_x[batch_counter, 0, :, :] = samples[0]
                    labels_x[batch_counter, class_num] = 1
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                for samples_num in range(len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[
                        samples_num
                    ]

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    counter += 1

            numeric_labels.append(pre_class)
        labels_w = [labels_x] + labels_yi
        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)
        labels_w_new = calculate_w(labels_w, n_way, num_shots)
        return_arr = [
            torch.from_numpy(batch_x),
            torch.from_numpy(labels_x),
            torch.from_numpy(labels_x_scalar),
            torch.from_numpy(labels_x_global),
            batches_xi,
            labels_yi,
            oracles_yi,
            labels_w_new,
        ]

        if cuda:
            return_arr = self.cast_cuda(return_arr)
        return return_arr


class Train_Generator_Aug(DataLoader):
    def __init__(self, root, keys=["0", "1"]):

        with open(root, "rb") as load_data:
            data_dict = pickle.load(load_data)
        data_ = {}
        for i in range(len(keys)):
            data_[i] = data_dict[keys[i]]
        self.data = data_
        self.channel = 1
        self.feature_shape = np.array((self.data[1][0])).shape

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(
        self,
        batch_size=5,
        n_way=2,
        num_shots=10,
        unlabeled_extra=0,
        cuda=True,
        variable=False,
    ):
        # init # features
        batch_x = np.zeros(
            (batch_size, self.channel, self.feature_shape[1], self.feature_shape[2]),
            dtype="float32",
        )
        # labels
        labels_x = np.zeros((batch_size, n_way), dtype="float32")
        labels_x_global = np.zeros(batch_size, dtype="int64")
        num = n_way * num_shots + 1

        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros(
                    (
                        batch_size,
                        self.channel,
                        self.feature_shape[1],
                        self.feature_shape[2],
                    ),
                    dtype="float32",
                )
            )
            labels_yi.append(np.zeros((batch_size, n_way), dtype="float32"))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype="float32")))

        # feed data

        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if (
                    class_num == pre_class
                ):  # 如果和随机选择的类别class_counter和随机产生的类号 positive_class
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    batch_x[batch_counter, 0, :, :] = samples[0]
                    labels_x[batch_counter, class_num] = 1
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                for samples_num in range(len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[
                        samples_num
                    ]

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    counter += 1

            numeric_labels.append(pre_class)
        labels_w = [labels_x] + labels_yi
        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)
        labels_w_new = calculate_w(labels_w, n_way, num_shots)
        return_arr = [
            torch.from_numpy(batch_x),
            torch.from_numpy(labels_x),
            torch.from_numpy(labels_x_scalar),
            torch.from_numpy(labels_x_global),
            batches_xi,
            labels_yi,
            oracles_yi,
            labels_w_new,
        ]

        if cuda:
            return_arr = self.cast_cuda(return_arr)
        return return_arr


class TestGenerator(DataLoader):
    def __init__(self, train_root, test_root, keys=["0", "1"]):
        # 读入训练数据
        with open(train_root, "rb") as load_data:
            train_data_dict = pickle.load(load_data)
        train_data_ = {}
        for i in range(len(keys)):
            train_data_[i] = train_data_dict[keys[i]]
        self.train_data = train_data_
        # 读入测试数据
        with open(test_root, "rb") as load_data:
            test_data_dict = pickle.load(load_data)
        test_data_ = {}
        for i in range(len(keys)):
            test_data_[i] = test_data_dict[keys[i]]
        self.test_data = test_data_

        self.channal = 1
        self.feature_shape = np.array((self.train_data[1][0])).shape

        slice_num_0 = 0
        slice_num_1 = 0
        slice_num = 0
        patient_slice_0 = []
        patient_slice_1 = []
        for k in range(len(self.test_data[0])):
            shape = np.array((self.test_data[0][k])).shape
            slice_num_0 += shape[0]
            patient_slice_0.append(shape[0])
        for k in range(len(self.test_data[1])):
            shape = np.array((self.test_data[1][k])).shape
            slice_num_1 += shape[0]
            patient_slice_1.append(shape[0])
        slice_num = slice_num_0 + slice_num_1

        self.slice_num_0 = slice_num_0
        self.slice_num_1 = slice_num_1
        self.slice_num = slice_num  # 切片的总数量
        self.patient_slice_0 = patient_slice_0  # 每个病人的切片数量
        self.patient_slice_1 = patient_slice_1

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_test_num(self):
        return self.slice_num, self.patient_slice_0, self.patient_slice_1

    def get_task_batch(
        self,
        pclass,
        patient,
        slice,
        batch_size=5,
        n_way=2,
        num_shots=5,
        unlabeled_extra=0,
        cuda=False,
        variable=False,
    ):
        # init
        batch_x = np.zeros(
            (batch_size, self.channal, self.feature_shape[1], self.feature_shape[2]),
            dtype="float32",
        )  # features
        labels_x = np.zeros((batch_size, n_way), dtype="float32")  # labels
        labels_x_global = np.zeros(batch_size, dtype="int64")
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros(
                    (
                        batch_size,
                        self.channal,
                        self.feature_shape[1],
                        self.feature_shape[2],
                    ),
                    dtype="float32",
                )
            )
            labels_yi.append(np.zeros((batch_size, n_way), dtype="float32"))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype="float32")))

        # feed data
        for batch_counter in range(batch_size):
            # 不随机选择类别，依次选择，确保每一个样本的每个切片都被测试到
            pre_class = pclass
            # pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if (
                    class_num == pre_class
                ):  # 如果和随机选择的类别class_counter和随机产生的类号 positive_class
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.train_data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape

                        select_num = np.random.randint(
                            0, shape_sample[0], 1
                        )  # 随机选择一张切片
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    # print(type(samples))
                    # Test sample
                    # print(np.array(samples[0]).shape)
                    # print(batch_x[batch_counter,0, :,:].shape)
                    batch_x[batch_counter, 0, :, :] = self.test_data[class_num][
                        patient
                    ][slice, :, :]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[0::]
                else:
                    samples = random.sample(self.train_data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape

                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    # print(type(samples))
                for samples_num in range(len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[
                        samples_num
                    ]
                    # except:
                    #     print(samples[samples_num])

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    # target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(pre_class)
        labels_w = [labels_x] + labels_yi
        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)
        labels_w_new = calculate_w(labels_w, n_way, num_shots)
        return_arr = [
            torch.from_numpy(batch_x),
            torch.from_numpy(labels_x),
            torch.from_numpy(labels_x_scalar),
            torch.from_numpy(labels_x_global),
            batches_xi,
            labels_yi,
            oracles_yi,
            labels_w_new,
        ]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        return return_arr


class Train_Generator_p(DataLoader):
    def __init__(self, root, p=1, keys=["0", "1"]):

        with open(root, "rb") as load_data:
            data_dict = pickle.load(load_data)
        data_ = {}
        for i in range(len(keys)):
            data_[i] = data_dict[keys[i]]

        # print(len(data_[0]),len(data_[1]))
        data_[0] = data_[0][: int(p * len(data_[0]))]
        data_[1] = data_[1][: int(p * len(data_[1]))]

        self.data = data_
        self.channel = 1
        self.feature_shape = np.array((self.data[1][0])).shape

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(
        self,
        batch_size=5,
        n_way=2,
        num_shots=10,
        unlabeled_extra=0,
        cuda=True,
        variable=False,
    ):
        # init # features
        batch_x = np.zeros(
            (batch_size, self.channel, self.feature_shape[1], self.feature_shape[2]),
            dtype="float32",
        )
        # labels
        labels_x = np.zeros((batch_size, n_way), dtype="float32")
        labels_x_global = np.zeros(batch_size, dtype="int64")
        num = n_way * num_shots + 1

        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros(
                    (
                        batch_size,
                        self.channel,
                        self.feature_shape[1],
                        self.feature_shape[2],
                    ),
                    dtype="float32",
                )
            )
            labels_yi.append(np.zeros((batch_size, n_way), dtype="float32"))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype="float32")))

        # feed data

        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if (
                    class_num == pre_class
                ):  # 如果和随机选择的类别class_counter和随机产生的类号 positive_class
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    batch_x[batch_counter, 0, :, :] = samples[0]
                    labels_x[batch_counter, class_num] = 1
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                for samples_num in range(len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[
                        samples_num
                    ]

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    counter += 1

            numeric_labels.append(pre_class)
        labels_w = [labels_x] + labels_yi
        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)
        labels_w_new = calculate_w(labels_w, n_way, num_shots)
        return_arr = [
            torch.from_numpy(batch_x),
            torch.from_numpy(labels_x),
            torch.from_numpy(labels_x_scalar),
            torch.from_numpy(labels_x_global),
            batches_xi,
            labels_yi,
            oracles_yi,
            labels_w_new,
        ]

        if cuda:
            return_arr = self.cast_cuda(return_arr)
        return return_arr


class Train_Generator_p_random(DataLoader):
    def __init__(self, root, p=1, keys=["0", "1"]):

        with open(root, "rb") as load_data:
            data_dict = pickle.load(load_data)
        data_ = {}
        for i in range(len(keys)):
            data_[i] = data_dict[keys[i]]

        random.shuffle(data_[0])
        random.shuffle(data_[1])

        # print(len(data_[0]),len(data_[1]))
        data_[0] = data_[0][: int(p * len(data_[0]))]
        data_[1] = data_[1][: int(p * len(data_[1]))]

        self.data = data_
        self.channel = 1
        self.feature_shape = np.array((self.data[1][0])).shape

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_task_batch(
        self,
        batch_size=5,
        n_way=2,
        num_shots=10,
        unlabeled_extra=0,
        cuda=True,
        variable=False,
    ):
        # init # features
        batch_x = np.zeros(
            (batch_size, self.channel, self.feature_shape[1], self.feature_shape[2]),
            dtype="float32",
        )
        # labels
        labels_x = np.zeros((batch_size, n_way), dtype="float32")
        labels_x_global = np.zeros(batch_size, dtype="int64")
        num = n_way * num_shots + 1

        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros(
                    (
                        batch_size,
                        self.channel,
                        self.feature_shape[1],
                        self.feature_shape[2],
                    ),
                    dtype="float32",
                )
            )
            labels_yi.append(np.zeros((batch_size, n_way), dtype="float32"))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype="float32")))

        # feed data
        for batch_counter in range(batch_size):
            pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if (
                    class_num == pre_class
                ):  # 如果和随机选择的类别class_counter和随机产生的类号 positive_class
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.data[class_num], num_shots + 1)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    batch_x[batch_counter, 0, :, :] = samples[0]
                    labels_x[batch_counter, class_num] = 1
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                for samples_num in range(len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[
                        samples_num
                    ]

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    counter += 1

            numeric_labels.append(pre_class)
        labels_w = [labels_x] + labels_yi
        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)
        labels_w_new = calculate_w(labels_w, n_way, num_shots)
        return_arr = [
            torch.from_numpy(batch_x),
            torch.from_numpy(labels_x),
            torch.from_numpy(labels_x_scalar),
            torch.from_numpy(labels_x_global),
            batches_xi,
            labels_yi,
            oracles_yi,
            labels_w_new,
        ]

        if cuda:
            return_arr = self.cast_cuda(return_arr)
        return return_arr


class TestGenerator_p(DataLoader):
    def __init__(self, train_root, test_root, p=1, keys=["0", "1"]):
        # 读入训练数据
        with open(train_root, "rb") as load_data:
            train_data_dict = pickle.load(load_data)
        train_data_ = {}
        for i in range(len(keys)):
            train_data_[i] = train_data_dict[keys[i]]

        train_data_[0] = train_data_[0][: int(p * len(train_data_[0]))]
        train_data_[1] = train_data_[1][: int(p * len(train_data_[1]))]

        self.train_data = train_data_
        # 读入测试数据
        with open(test_root, "rb") as load_data:
            test_data_dict = pickle.load(load_data)
        test_data_ = {}
        for i in range(len(keys)):
            test_data_[i] = test_data_dict[keys[i]]

        test_data_[0] = test_data_[0][: int(p * len(test_data_[0]))]
        test_data_[1] = test_data_[1][: int(p * len(test_data_[1]))]

        self.test_data = test_data_

        self.channal = 1
        self.feature_shape = np.array((self.train_data[1][0])).shape

        slice_num_0 = 0
        slice_num_1 = 0
        slice_num = 0
        patient_slice_0 = []
        patient_slice_1 = []
        for k in range(len(self.test_data[0])):
            shape = np.array((self.test_data[0][k])).shape
            slice_num_0 += shape[0]
            patient_slice_0.append(shape[0])
        for k in range(len(self.test_data[1])):
            shape = np.array((self.test_data[1][k])).shape
            slice_num_1 += shape[0]
            patient_slice_1.append(shape[0])
        slice_num = slice_num_0 + slice_num_1

        self.slice_num_0 = slice_num_0
        self.slice_num_1 = slice_num_1
        self.slice_num = slice_num  # 切片的总数量
        self.patient_slice_0 = patient_slice_0  # 每个病人的切片数量
        self.patient_slice_1 = patient_slice_1

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_test_num(self):
        return self.slice_num, self.patient_slice_0, self.patient_slice_1

    def get_task_batch(
        self,
        pclass,
        patient,
        slice,
        batch_size=5,
        n_way=2,
        num_shots=5,
        unlabeled_extra=0,
        cuda=False,
        variable=False,
    ):
        # init
        batch_x = np.zeros(
            (batch_size, self.channal, self.feature_shape[1], self.feature_shape[2]),
            dtype="float32",
        )  # features
        labels_x = np.zeros((batch_size, n_way), dtype="float32")  # labels
        labels_x_global = np.zeros(batch_size, dtype="int64")
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros(
                    (
                        batch_size,
                        self.channal,
                        self.feature_shape[1],
                        self.feature_shape[2],
                    ),
                    dtype="float32",
                )
            )
            labels_yi.append(np.zeros((batch_size, n_way), dtype="float32"))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype="float32")))

        # feed data
        for batch_counter in range(batch_size):
            # 不随机选择类别，依次选择，确保每一个样本的每个切片都被测试到
            pre_class = pclass
            # pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if (
                    class_num == pre_class
                ):  # 如果和随机选择的类别class_counter和随机产生的类号 positive_class
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.train_data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape

                        select_num = np.random.randint(
                            0, shape_sample[0], 1
                        )  # 随机选择一张切片
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    # print(type(samples))
                    # Test sample
                    # print(np.array(samples[0]).shape)
                    # print(batch_x[batch_counter,0, :,:].shape)
                    batch_x[batch_counter, 0, :, :] = self.test_data[class_num][
                        patient
                    ][slice, :, :]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[0::]
                else:
                    samples = random.sample(self.train_data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape

                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    # print(type(samples))
                for samples_num in range(len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[
                        samples_num
                    ]
                    # except:
                    #     print(samples[samples_num])

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    # target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(pre_class)
        labels_w = [labels_x] + labels_yi
        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)
        labels_w_new = calculate_w(labels_w, n_way, num_shots)
        return_arr = [
            torch.from_numpy(batch_x),
            torch.from_numpy(labels_x),
            torch.from_numpy(labels_x_scalar),
            torch.from_numpy(labels_x_global),
            batches_xi,
            labels_yi,
            oracles_yi,
            labels_w_new,
        ]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        return return_arr


class TestGenerator_p_random(DataLoader):
    def __init__(self, train_root, test_root, p=1, keys=["0", "1"]):
        # 读入训练数据
        with open(train_root, "rb") as load_data:
            train_data_dict = pickle.load(load_data)
        train_data_ = {}
        for i in range(len(keys)):
            train_data_[i] = train_data_dict[keys[i]]

        random.shuffle(train_data_[0])
        random.shuffle(train_data_[1])

        train_data_[0] = train_data_[0][: int(p * len(train_data_[0]))]
        train_data_[1] = train_data_[1][: int(p * len(train_data_[1]))]

        self.train_data = train_data_
        # 读入测试数据
        with open(test_root, "rb") as load_data:
            test_data_dict = pickle.load(load_data)
        test_data_ = {}
        for i in range(len(keys)):
            test_data_[i] = test_data_dict[keys[i]]

        random.shuffle(test_data_[0])
        random.shuffle(test_data_[1])

        test_data_[0] = test_data_[0][: int(p * len(test_data_[0]))]
        test_data_[1] = test_data_[1][: int(p * len(test_data_[1]))]

        self.test_data = test_data_

        self.channal = 1
        self.feature_shape = np.array((self.train_data[1][0])).shape

        slice_num_0 = 0
        slice_num_1 = 0
        slice_num = 0
        patient_slice_0 = []
        patient_slice_1 = []
        for k in range(len(self.test_data[0])):
            shape = np.array((self.test_data[0][k])).shape
            slice_num_0 += shape[0]
            patient_slice_0.append(shape[0])
        for k in range(len(self.test_data[1])):
            shape = np.array((self.test_data[1][k])).shape
            slice_num_1 += shape[0]
            patient_slice_1.append(shape[0])
        slice_num = slice_num_0 + slice_num_1

        self.slice_num_0 = slice_num_0
        self.slice_num_1 = slice_num_1
        self.slice_num = slice_num  # 切片的总数量
        self.patient_slice_0 = patient_slice_0  # 每个病人的切片数量
        self.patient_slice_1 = patient_slice_1

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def get_test_num(self):
        return self.slice_num, self.patient_slice_0, self.patient_slice_1

    def get_task_batch(
        self,
        pclass,
        patient,
        slice,
        batch_size=5,
        n_way=2,
        num_shots=5,
        unlabeled_extra=0,
        cuda=False,
        variable=False,
    ):
        # init
        batch_x = np.zeros(
            (batch_size, self.channal, self.feature_shape[1], self.feature_shape[2]),
            dtype="float32",
        )  # features
        labels_x = np.zeros((batch_size, n_way), dtype="float32")  # labels
        labels_x_global = np.zeros(batch_size, dtype="int64")
        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []
        for i in range(n_way * num_shots):
            batches_xi.append(
                np.zeros(
                    (
                        batch_size,
                        self.channal,
                        self.feature_shape[1],
                        self.feature_shape[2],
                    ),
                    dtype="float32",
                )
            )
            labels_yi.append(np.zeros((batch_size, n_way), dtype="float32"))
            oracles_yi.append((np.zeros((batch_size, n_way), dtype="float32")))

        # feed data
        for batch_counter in range(batch_size):
            # 不随机选择类别，依次选择，确保每一个样本的每个切片都被测试到
            pre_class = pclass
            # pre_class = random.randint(0, n_way - 1)
            indexes_perm = np.random.permutation(n_way * num_shots)
            counter = 0
            for class_num in range(0, n_way):
                if (
                    class_num == pre_class
                ):  # 如果和随机选择的类别class_counter和随机产生的类号 positive_class
                    # We take num_shots + one sample for one class
                    samples = random.sample(self.train_data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape

                        select_num = np.random.randint(
                            0, shape_sample[0], 1
                        )  # 随机选择一张切片
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    # print(type(samples))
                    # Test sample
                    # print(np.array(samples[0]).shape)
                    # print(batch_x[batch_counter,0, :,:].shape)
                    batch_x[batch_counter, 0, :, :] = self.test_data[class_num][
                        patient
                    ][slice, :, :]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[0::]
                else:
                    samples = random.sample(self.train_data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape

                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :, :])
                    samples = samples_
                    # print(type(samples))
                for samples_num in range(len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :] = samples[
                        samples_num
                    ]
                    # except:
                    #     print(samples[samples_num])

                    labels_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_num] = 1
                    # target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(pre_class)
        labels_w = [labels_x] + labels_yi
        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_x_scalar = np.argmax(labels_x, 1)
        labels_w_new = calculate_w(labels_w, n_way, num_shots)
        return_arr = [
            torch.from_numpy(batch_x),
            torch.from_numpy(labels_x),
            torch.from_numpy(labels_x_scalar),
            torch.from_numpy(labels_x_global),
            batches_xi,
            labels_yi,
            oracles_yi,
            labels_w_new,
        ]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        return return_arr
