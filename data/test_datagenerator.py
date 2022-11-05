# -!- coding: utf-8 -!-

import pickle
import random

import numpy as np
import torch
import torch.utils.data as data

from data.datagenerator import calculate_w


class TestGenerator(data.DataLoader):
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

        self.channel = 1
        self.feature_shape = np.array((self.train_data[1][0])).shape

        slice_num_0 = 0
        slice_num_1 = 0
        patient_slice_0 = []
        patient_slice_1 = []
        print("==>", (len(self.test_data[0])))
        print("==>", (len(self.test_data[1])))
        # input()
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
        p_class,
        patient,
        slice,
        batch_size=5,
        n_way=2,
        num_shots=10,
        unlabeled_extra=0,
        cuda=False,
        variable=False,
    ):
        # init
        batch_x = np.zeros(
            (batch_size, self.channel, self.feature_shape[1], self.feature_shape[2]),
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
            # 不随机选择类别，依次选择，确保每一个样本的每个切片都被测试到
            pre_class = p_class
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
                        samples_.append(samples[samples_num][select_num, :])
                    samples = samples_
                    # print(type(samples))
                    # Test sample
                    # print(np.array(samples[0]).shape)
                    # print(batch_x[batch_counter,0, :,:].shape)
                    batch_x[batch_counter, 0, :, :] = self.test_data[class_num][
                        patient
                    ][slice, :]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[0::]
                else:
                    samples = random.sample(self.train_data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = samples[samples_num].shape

                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :])
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
        return_arr = self.cast_cuda(return_arr)
        return return_arr


if __name__ == "__main__":
    loadData = np.load("data//data_person.npy", allow_pickle=True)
    loadLabel = np.load("data//label_person.npy", allow_pickle=True)
    print("----type----")
    print(type(loadData))
    print("----shape----")
    slice = 0
    for i in range(72):
        print(loadData[i].shape)
        slice += loadData[i].shape[0]
        print("max", np.max(loadData[i]))
        print("min", np.min(loadData[i]))
    print("total slice:", slice)
    print(loadData.shape)

    print("====================")
    print("----type----")
    print(type(loadLabel))
    print("----shape----")
    print(loadLabel.shape)
