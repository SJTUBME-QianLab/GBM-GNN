#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2022/1/24 0:00
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : GNN.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 这个文件用于生成用到的网络模型，一层又一层，主要是模型的选择
# 对每一幅图像做常规卷积操作 得到图像的隐藏状态，也即在另一个空间的表示形式，而已 return [e1, e2, e3, output]
from models.gnn_iclr import GNN_nl, GNN_active, GNN_nl_omniglot


class EmbeddingOmniglot(nn.Module):
    """ In this network the input image is supposed to be 28x28 """

    def __init__(self, args, emb_size):
        super(EmbeddingOmniglot, self).__init__()
        self.emb_size = emb_size
        self.nef = 64
        self.args = args

        # input is 1 x 28 x 28
        self.conv1 = nn.Conv2d(
            1, self.nef, 3, padding=1, bias=False
        )  # 输入有1个通道，输出有64个通道，
        # 卷积核尺寸为3，卷积步长为默认，无偏置
        # padding为输入的每一条边补充的0的层数，保持图片大小不变
        self.bn1 = nn.BatchNorm2d(self.nef)  # 对小批量(mini-batch)3d数据组成的4d输入
        #  进行批标准化(Batch Normalization)操作
        # state size. (nef) x 14 x 14
        self.conv2 = nn.Conv2d(self.nef, self.nef, 3, padding=1, bias=False)  # 又一个卷积层
        self.bn2 = nn.BatchNorm2d(self.nef)  # 又一次标准化

        # state size. (1.5*ndf) x 7 x 7
        self.conv3 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 5 x 5
        self.conv4 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn4 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 3 x 3
        self.fc_last = nn.Linear(
            3 * 3 * self.nef, self.emb_size, bias=False
        )  # 对输入数据做线性变换
        self.bn_last = nn.BatchNorm1d(
            self.emb_size
        )  # 对emb_size标准化，至于emb_size是feature的大小

    def forward(self, inputs):
        # print(inputs.shape)
        # input('=++++++++++++++++++++++')
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs)), 2)  # input 是什么格式的数据呢？？？？？？
        x = F.leaky_relu(e1, 0.1, inplace=True)

        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.1, inplace=True)

        e3 = self.bn3(self.conv3(x))
        x = F.leaky_relu(e3, 0.1, inplace=True)
        e4 = self.bn4(self.conv4(x))
        x = F.leaky_relu(e4, 0.1, inplace=True)
        x = x.view(-1, 3 * 3 * self.nef)  # x是最终得到的每个元素的特征的集合，特征大小为3*3*self.nef

        output = F.leaky_relu(self.bn_last(self.fc_last(x)))  # 线性变换再激活后得到最终的输出

        return [e1, e2, e3, output]


# return [e1, e2, e3, e4, None, output]


class EmbeddingImagenet(nn.Module):
    """ In this network the input image is supposed to be 28x28 """

    def __init__(self, args, emb_size):
        super(EmbeddingImagenet, self).__init__()
        self.emb_size = emb_size
        self.ndf = 16
        self.args = args

        # Input 84x84x3
        self.conv1 = nn.Conv2d(
            1, self.ndf, kernel_size=3, stride=1, padding=1, bias=False
        )  # !!!!!!!!!!!!!!!!!!!!
        self.bn1 = nn.BatchNorm2d(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf * 1.5), kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.ndf * 1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2d(
            int(self.ndf * 1.5), self.ndf * 2, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.ndf * 2)
        self.drop_3 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2d(
            self.ndf * 2, self.ndf * 4, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(self.ndf * 4)
        self.drop_4 = nn.Dropout2d(0.5)

        # Input 5x5x256
        self.fc1 = nn.Linear(self.ndf * 4 * 13 * 11, self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2, inplace=True)
        # print('1 layer:',x.shape)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        # print('2 layer:', x.shape)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        # print('3 layer:', x.shape)
        x = self.drop_3(x)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        # print('4 layer:', x.shape)
        # x = x.view(-1, self.ndf*4*5*5)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x = x.view(-1, self.ndf * 4 * 13 * 11)
        output = self.bn_fc(self.fc1(x))

        return [e1, e2, e3, e4, None, output]


# 使用定义好的网络计算结果
class MetricNN(nn.Module):
    def __init__(self, args, emb_size):
        super(MetricNN, self).__init__()

        self.metric_network = args.metric_network
        self.emb_size = emb_size  # 表示一幅图像的向量维度
        self.args = args

        if self.metric_network == "gnn_iclr_nl":
            assert self.args.train_N_way == self.args.test_N_way
            num_inputs = self.emb_size + self.args.train_N_way
            if self.args.dataset == "mini_imagenet":  # it's it
                self.gnn_obj = GNN_nl(args, num_inputs, nf=96, J=1)
            elif "omniglot" in self.args.dataset:
                self.gnn_obj = GNN_nl_omniglot(args, num_inputs, nf=96, J=1)
        elif self.metric_network == "gnn_iclr_active":
            assert self.args.train_N_way == self.args.test_N_way
            num_inputs = self.emb_size + self.args.train_N_way
            self.gnn_obj = GNN_active(args, num_inputs, 96, J=1)
        else:
            raise NotImplementedError

    def gnn_iclr_forward(self, z, zi_s, labels_yi):
        # Creating WW matrix
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))  # batch_size
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)

        # print(nodes.size()) # 16*11*258

        W_ALL, logits = self.gnn_obj(nodes)
        # logits: torch.Size([16, 2])
        logits = logits.squeeze(-1)
        outputs = F.sigmoid(logits)

        return W_ALL, outputs, logits

    def gnn_iclr_active_forward(self, z, zi_s, labels_yi, oracles_yi, hidden_layers):
        # Creating WW matrix
        zero_pad = Variable(
            torch.ones(labels_yi[0].size()) * 1.0 / labels_yi[0].size(1)
        )
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [torch.cat([label_yi, zi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)

        oracles_yi = [zero_pad] + oracles_yi
        oracles_yi = [oracle_yi.unsqueeze(1) for oracle_yi in oracles_yi]
        oracles_yi = torch.cat(oracles_yi, 1)

        logits = self.gnn_obj(nodes, oracles_yi, hidden_layers).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, logits

    def forward(self, inputs):
        """input: [batch_x, [batches_xi], [labels_yi]]"""
        [z, zi_s, labels_yi, oracles_yi] = inputs

        if "gnn_iclr_active" in self.metric_network:
            return self.gnn_iclr_active_forward(z, zi_s, labels_yi, oracles_yi)
        elif "gnn_iclr" in self.metric_network:
            return self.gnn_iclr_forward(z, zi_s, labels_yi)
        else:
            raise NotImplementedError


class SoftmaxModule:
    def __init__(self):
        self.softmax_metric = "log_softmax"

    def forward(self, outputs):
        if self.softmax_metric == "log_softmax":
            return F.log_softmax(outputs)
        else:
            raise (NotImplementedError)


def load_model(model_name, args, io):
    try:
        model = torch.load("checkpoints/%s/models/%s.t7" % (args.exp_name, model_name))
        io.cprint("Loading Parameters from the last trained %s Model" % model_name)
        return model
    except:
        io.cprint("Initiallize new Network Weights for %s" % model_name)
        pass
    return None


def create_models(args):
    print(args.dataset)

    if "omniglot" == args.dataset:
        enc_nn = EmbeddingOmniglot(args, 64)
    elif "mini_imagenet" == args.dataset:
        enc_nn = EmbeddingImagenet(args, 256)
    else:
        raise NameError("Dataset " + args.dataset + " not knows")
    return enc_nn, MetricNN(args, emb_size=256)
