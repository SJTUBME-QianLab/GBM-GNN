from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import random


class JCD_Dataset(Dataset):
    def __init__(self, dataset, label, iteration=10000):

        self.data = dataset
        self.label = label
        self.iteration = iteration

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

    def __getitem__(self, index):
        # 按照序号index返回数据和标签
        if index % 2 == 0:  # 无放回抽取一个0类
            length = len(self.label0_index_box)  # 获取一下盒子中剩余元素的个数
            if length == 0:  # 如果盒子为空，则重新填满
                self.label0_index_box = np.arange(len(self.label_score0))
                length = len(self.label0_index_box)
            random_index = random.randint(0, length - 1)  # 从盒子中剩余的样本中随机抽取一个
            sk = self.sk_data_score0[self.label0_index_box[random_index]]
            y = self.label_score0[self.label0_index_box[random_index]]
            self.label0_index_box = np.delete(
                self.label0_index_box, random_index
            )  # 将抽出的样本序号从盒子中删除
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


class my_Test(Dataset):
    def __init__(self, fold=0):
        self.dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __getitem__(self, index):
        # 按照序号index返回数据和标签
        print(index)
        return self.dataset[index]

    def __len__(self):
        # 返回数据库长度
        return 10


if __name__ == "__main__":
    dset_train = my_Test(fold=0)
    train_loader = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=0)
    for train_data in train_loader:
        print(train_data)
