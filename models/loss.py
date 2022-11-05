#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 22:47
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : loss.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080
import torch
import torch.nn.functional as F


class w_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w_all, labels_w, loss, weight=0.001):
        w_loss = 0
        for i in range(len(w_all)):
            w_loss += torch.norm((labels_w) - (w_all[i]), p=1)
        # print("W Loss:", w_loss)
        return loss + weight * w_loss


def adjust_learning_rate(optimizers, lr, iter, args):
    new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


class SoftmaxModule:
    def __init__(self):
        self.softmax_metric = "log_softmax"

    def forward(self, outputs):
        if self.softmax_metric == "log_softmax":
            return F.log_softmax(outputs)
        else:
            raise (NotImplementedError)


class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w_all, labels_w, loss, weight=0.001):
        w_loss = 0
        for i in range(len(w_all)):
            w_loss += torch.norm((labels_w) - (w_all[i]), p=1)
        return loss + weight * w_loss
