# -!- coding: utf-8 -!-

import pickle
import random
import numpy as np
import torch
import torch.utils.data as data


def calculate_w(labels_w, n_way, num_shots):
    num = n_way * num_shots + 1
    labels_w = np.array(labels_w)[:, :, 0]
    labels_w = torch.from_numpy(labels_w)
    labels_w = torch.transpose(labels_w, 0, 1)
    labels_w = labels_w.unsqueeze(2)
    labels_w1 = torch.transpose(labels_w, 1, 2)
    labels_w_new = torch.abs(torch.abs(labels_w - labels_w1) - 1) - torch.eye(num)
    return labels_w_new


class Generator(data.DataLoader):
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

    def get_task_batch(self, batch_size=5, n_way=2, num_shots=10):
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
                        shape_sample = np.array(samples[samples_num]).shape
                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :])
                    samples = samples_
                    batch_x[batch_counter, 0, :, :] = samples[0]
                    labels_x[batch_counter, class_num] = 1  # one hot
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_num], num_shots)
                    samples_ = []
                    for samples_num in range(len(samples)):
                        shape_sample = np.array(samples[samples_num]).shape

                        select_num = np.random.randint(0, shape_sample[0], 1)
                        samples_.append(samples[samples_num][select_num, :])
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

    LOADER = Generator(root="data//psp_data.pkl")

    test = LOADER.get_task_batch(
        batch_size=5,
        n_way=2,
        num_shots=10,
    )
