#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Pytorch requirements
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# 生成基本的神经网络模块
from models.cbam import CBAM

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor

# 一种运算  节点特征的运算
# x is a tensor of size (bs, N, num_features)  #图有N个节点，每个节点有num_features维的特征表达
# W is a tensor of size (bs, N, N, J)          # J表征节点之间的联系
# output has size (bs, N, J*num_features)


def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)  #图有N个节点，每个节点有num_features维的特征表达
    # W is a tensor of size (bs, N, N, J)          #猜测 j 是类似卷积核个数的一个东西？？？？？？？ j改变了特征的维度

    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]  # N个节点
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3)  # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x)  # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2)  # output has size (bs, N, J*num_features)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J * nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):  # x对应的是图，W对应的是连接矩阵  未对W做改动，只是读出后照原样返回
        W = input[0]
        x = gmul(input)  # out has size (bs, N, num_inputs)
        # print(x.size())
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)  # 重新整理x的结构
        x = self.fc(x)  # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x


# 学习图中节点之间的相互关系 即论文中的 A
class Wcompute(nn.Module):
    def __init__(
        self,
        input_features,
        nf,
        operator="J2",
        activation="softmax",
        ratio=[2, 2, 1, 1],
        num_operators=1,
        drop=False,
        args="",
    ):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf * ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf * ratio[2])
        self.conv2d_4 = nn.Conv2d(nf * ratio[2], nf * ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf * ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation
        self.att_type = args.att_type
        if args.att_type == "CBAM":
            self.cbam_1 = CBAM(int(nf * ratio[0]), 16)
            # self.cbam_2 = CBAM(int(nf * ratio[1]), 16)
            # self.cbam_3 = CBAM(nf * ratio[2], 16)
            # self.cbam_4 = CBAM(nf * ratio[3], 16)

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)  # W1 size: bs x N x 1 x num_features
        W2 = torch.transpose(W1, 1, 2)  # W2 size: bs x 1 x N x num_features
        W_new = torch.abs(W1 - W2)  # W_new size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3)  # size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)

        if self.att_type == "CBAM":
            residual = W_new
            out = self.cbam_1(W_new)
            W_new = out + residual

        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)
        # if self.att_type == 'CBAM':
        #     residual = W_new
        #     out = self.cbam_2(W_new)
        #     W_new = out + residual
        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)
        # if self.att_type == 'CBAM':
        #     residual = W_new
        #     out = self.cbam_3(W_new)
        #     W_new = out + residual
        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)
        # if self.att_type == 'CBAM':
        #     residual = W_new
        #     out = self.cbam_4(W_new)
        #     W_new = out + residual
        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3)  # size: bs x N x N x 1

        if self.activation == "softmax":
            W_new = W_new - W_id.expand_as(W_new) * 1e8  # 将W_id 扩充到 W_new 的维度
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == "sigmoid":
            W_new = F.sigmoid(W_new)
            W_new *= 1 - W_id
        elif self.activation == "none":
            W_new *= 1 - W_id
        else:
            raise (NotImplementedError)

        if self.operator == "laplace":
            W_new = W_id - W_new
        elif self.operator == "J2":
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise (NotImplementedError)

        return W_new


class GNN_nl(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        if args.dataset == "mini_imagenet":
            self.num_layers = 2
        else:
            self.num_layers = 2

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(
                    self.input_features,
                    nf,
                    operator="J2",
                    activation="softmax",
                    ratio=[2, 2, 1, 1],
                    args=args,
                )
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(
                    self.input_features + int(nf / 2) * i,
                    nf,
                    operator="J2",
                    activation="softmax",
                    ratio=[2, 2, 1, 1],
                    args=args,
                )
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module("layer_w{}".format(i), module_w)
            self.add_module("layer_l{}".format(i), module_l)

        self.w_comp_last = Wcompute(
            self.input_features + int(self.nf / 2) * self.num_layers,
            nf,
            operator="J2",
            activation="softmax",
            ratio=[2, 2, 1, 1],
            args=args,
        )
        self.layer_last = Gconv(
            self.input_features + int(self.nf / 2) * self.num_layers,
            args.train_N_way,
            2,
            bn_bool=False,
        )

    def forward(self, x):
        # print(x.size())
        W_init = Variable(
            torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)
        )
        # 维度为batch_size*(n_way*num_shot+1)*(n_way*num_shot+1)*1
        W_ALL = []
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules["layer_w{}".format(i)](x, W_init)
            W_ALL.append(Wi[:, :, :, 1])
            x_new = F.leaky_relu(self._modules["layer_l{}".format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]  # 这个[1]值得是取第二个返回值x  ps：gcov有两个返回值
        # out: torch.Size([16, 11, 2])
        W_ALL.append(Wl[:, :, :, 1])
        # print(self.layer_last.num_inputs,self.layer_last.num_outputs)
        return W_ALL, out[:, 0, :]


class GNN_nl_omniglot(nn.Module):  # 整个的一个网络的结构，一层又一层
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl_omniglot, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2
        for i in range(self.num_layers):
            module_w = Wcompute(
                self.input_features + int(nf / 2) * i,
                self.input_features + int(nf / 2) * i,
                operator="J2",
                activation="softmax",
                ratio=[2, 1.5, 1, 1],
                drop=False,
                args=args,
            )
            module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module("layer_w{}".format(i), module_w)
            self.add_module("layer_l{}".format(i), module_l)

        self.w_comp_last = Wcompute(
            self.input_features + int(self.nf / 2) * self.num_layers,
            self.input_features + int(self.nf / 2) * (self.num_layers - 1),
            operator="J2",
            activation="softmax",
            ratio=[2, 1.5, 1, 1],
            drop=True,
            args=args,
        )
        self.layer_last = Gconv(
            self.input_features + int(self.nf / 2) * self.num_layers,
            args.train_N_way,
            2,
            bn_bool=True,
        )

    def forward(self, x):
        W_init = Variable(
            torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)
        )
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules["layer_w{}".format(i)](x, W_init)

            x_new = F.leaky_relu(self._modules["layer_l{}".format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]


class GNN_active(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_active, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2
        for i in range(self.num_layers // 2):
            if i == 0:
                module_w = Wcompute(
                    self.input_features,
                    nf,
                    operator="J2",
                    activation="softmax",
                    ratio=[2, 2, 1, 1],
                    args=args,
                )
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(
                    self.input_features + int(nf / 2) * i,
                    nf,
                    operator="J2",
                    activation="softmax",
                    ratio=[2, 2, 1, 1],
                    args=args,
                )
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)

            self.add_module("layer_w{}".format(i), module_w)
            self.add_module("layer_l{}".format(i), module_l)

        self.conv_active_1 = nn.Conv1d(
            self.input_features + int(nf / 2) * 1,
            self.input_features + int(nf / 2) * 1,
            1,
        )
        self.bn_active = nn.BatchNorm1d(self.input_features + int(nf / 2) * 1)
        self.conv_active_2 = nn.Conv1d(self.input_features + int(nf / 2) * 1, 1, 1)

        for i in range(int(self.num_layers / 2), self.num_layers):
            if i == 0:
                module_w = Wcompute(
                    self.input_features,
                    nf,
                    operator="J2",
                    activation="softmax",
                    ratio=[2, 2, 1, 1],
                    args=args,
                )
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(
                    self.input_features + int(nf / 2) * i,
                    nf,
                    operator="J2",
                    activation="softmax",
                    ratio=[2, 2, 1, 1],
                    args=args,
                )
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module("layer_w{}".format(i), module_w)
            self.add_module("layer_l{}".format(i), module_l)

        self.w_comp_last = Wcompute(
            self.input_features + int(self.nf / 2) * self.num_layers,
            nf,
            operator="J2",
            activation="softmax",
            ratio=[2, 2, 1, 1],
            args=args,
        )
        self.layer_last = Gconv(
            self.input_features + int(self.nf / 2) * self.num_layers,
            args.train_N_way,
            2,
            bn_bool=False,
        )

    def active(self, x, oracles_yi, hidden_labels):
        x_active = torch.transpose(x, 1, 2)
        x_active = self.conv_active_1(x_active)
        x_active = F.leaky_relu(self.bn_active(x_active))
        x_active = self.conv_active_2(x_active)
        x_active = torch.transpose(x_active, 1, 2)

        x_active = x_active.squeeze(-1)
        x_active = x_active - (1 - hidden_labels) * 1e8
        x_active = F.softmax(x_active)
        x_active = x_active * hidden_labels

        if self.args.active_random == 1:
            # print('random active')
            x_active.data.fill_(1.0 / x_active.size(1))
            decision = torch.multinomial(x_active)
            x_active = x_active.detach()
        else:
            if self.training:
                decision = torch.multinomial(x_active)
            else:
                _, decision = torch.max(x_active, 1)
                decision = decision.unsqueeze(-1)

        decision = decision.detach()

        mapping = torch.FloatTensor(decision.size(0), x_active.size(1)).zero_()
        mapping = Variable(mapping)
        if self.args.cuda:
            mapping = mapping.cuda()
        mapping.scatter_(1, decision, 1)

        mapping_bp = (x_active * mapping).unsqueeze(-1)
        mapping_bp = mapping_bp.expand_as(oracles_yi)

        label2add = mapping_bp * oracles_yi  # bsxNodesxN_way
        padd = torch.zeros(x.size(0), x.size(1), x.size(2) - label2add.size(2))
        padd = Variable(padd).detach()
        if self.args.cuda:
            padd = padd.cuda()
        label2add = torch.cat([label2add, padd], 2)

        x = x + label2add
        return x

    def forward(self, x, oracles_yi, hidden_labels):
        W_init = Variable(
            torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)
        )
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers // 2):
            Wi = self._modules["layer_w{}".format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules["layer_l{}".format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        x = self.active(x, oracles_yi, hidden_labels)

        for i in range(int(self.num_layers / 2), self.num_layers):
            Wi = self._modules["layer_w{}".format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules["layer_l{}".format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]


if __name__ == "__main__":
    # cam_result modules
    bs = 4
    nf = 10
    num_layers = 5
    num_features = nf
    N = 7
    x = torch.ones((bs, N, nf))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### cam_result gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out.shape)
    # print(out[0, :, num_features:])
    ######################### cam_result gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### cam_result gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())
