#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 20:00
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : pre_train_resnet.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
)
parser.add_argument(
    "--model", type=str, default="ResNetCBAM", help="model (ResNet, ResNetCBAM)"
)
parser.add_argument("--train_type", type=str, default="balance")
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="batch_size", help="Size of batch)"
)
parser.add_argument(
    "--iteration", type=int, default=80000, metavar="iter", help="training iteration)"
)
parser.add_argument(
    "--inter", type=int, default=200, metavar="inter", help="print iters)"
)
parser.add_argument("--r", type=int, default=340, metavar="seed", help="random seed")
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    metavar="LR",
    help="learning rate (default: 0.001)",
)
parser.add_argument(
    "--weight-decay",
    type=int,
    default=0.0005,
    metavar="N",
    help="weight-decay (default: 0.0005)",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    metavar="N",
    help="input visible devices for training (default: 0)",
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.data import JCD_Dataset, MyDataset
from models.mynet import resnet18
from models.mynet_dirk_full import resnet18_dirk_full
from utils import io_utils
from utils.device import func_device
from utils.util import (
    adjust_learning_rate_D,
    setup_seed,
    get_time,
    new_path,
    paint_in_pre_train,
    save_arg,
)

args.device = func_device()
args.cuda = not args.no_cuda and torch.cuda.is_available()
setup_seed()


def fold_(acc_5, io):
    acc_5_np = np.array(acc_5)

    average = acc_5_np.mean(axis=0)
    average = average.tolist()

    for n in range(len(average)):
        io.cprint("================================")
        io.cprint(str(n) + "====" + str(average[n]))
        for i in range(5):
            io.cprint(str(acc_5[i][n]))

    index_h_acc = average.index(max(average))
    io.cprint("index_h_acc" + str(index_h_acc))
    io.cprint("the highest average acc is:" + str(average[index_h_acc]))
    io.cprint("==============================")
    for i in range(5):
        io.cprint(str(acc_5[i][index_h_acc]))


def data_loaders(data_dir, batch_size, normal=False, balance=True, iteration=10000):
    with open(data_dir, "rb") as f:
        data = pickle.load(f)
    img_slice_data = []
    labels = []
    person = []
    p = 0
    for k in data.keys():
        for nii in data[k]:
            p += 1
            for slice in range(nii.shape[0]):
                img_slice_data.append(nii[slice])
                labels.append(int(k))
                person.append(p)

    if balance:
        train_dataloader = DataLoader(
            JCD_Dataset(img_slice_data, labels, normal, iteration=iteration),
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
        )
        return train_dataloader

    train_dataloader = DataLoader(
        MyDataset(img_slice_data, labels, person, normal),
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
    )
    return train_dataloader


def person_acc(pre, real, person):
    person_0 = 0
    person_1 = 0
    person_0_pre_ture = 0
    person_1_pre_ture = 0
    for p in set(person):
        location = [person[index_] == p for index_ in range(len(person))]
        person_pre = np.array(pre)[location]
        person_real = np.array(real)[location]
        if person_real[0] == 1:
            person_1 += 1
            if abs(sum(person_pre - person_real)) <= len(person_pre) / 2:
                person_1_pre_ture += 1
        if person_real[0] == 0:
            person_0 += 1
            if abs(sum(person_pre - person_real)) <= len(person_pre) / 2:
                person_0_pre_ture += 1
    return (
        ("1:{}//{}".format(person_1_pre_ture, person_1)),
        ("0:{}//{}".format(person_0_pre_ture, person_0)),
    )


if __name__ == "__main__":
    save_model = True
    test_amp = 0
    cur_time = "r{}".format(args.r)
    classes = [0, 1]
    net_type = "{}_bt_{}_iter_{}_testamp_{}_pre_train_{}_45_{}".format(
        args.model, args.batch_size, args.iteration, test_amp, args.train_type, cur_time
    )
    path = new_path("checkpoints/" + net_type + "/")
    io = io_utils.IOStream(path + "save_result_run_person_test.log")
    acc_5 = []
    save_arg(args, join(path + "save_result_run_person_test.log"))
    for fold in range(5):
        train_loss_, test_loss_y, test_acc_0, test_acc_1, test_acc_y = (
            [],
            [],
            [],
            [],
            [],
        )
        data_dir = "../processed_data/amp_balance_enlarge_train_{}_45_r{}.pkl".format(
            str(fold), args.r
        )
        train_dataloader = data_loaders(
            data_dir, batch_size=args.batch_size, iteration=args.iteration
        )
        data_dir_test = "../processed_data/only_enlarge_test_{}_45_r{}.pkl".format(
            str(fold), args.r
        )
        test_dataloader = data_loaders(
            data_dir_test, batch_size=args.batch_size, balance=False
        )

        if args.model == "ResNet":
            cnn = resnet18()
        elif args.model == "ResNetCBAM":
            cnn = resnet18_dirk_full()

        cnn = cnn.cuda()
        weight = torch.from_numpy(np.array([0.2, 0.8])).float().cuda()
        criterion = nn.CrossEntropyLoss(weight=weight)
        cnn_optimizer = optim.SGD(
            cnn.parameters(), lr=args.lr, momentum=0.99, weight_decay=args.weight_decay
        )

        # track change in validation loss
        valid_loss_min = np.Inf
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        cnn.train()
        iteration = 0
        for data, target in train_dataloader:

            data = data.cuda()
            target = target.cuda()

            cnn_optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = cnn(data)
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            cnn_optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

            iteration += 1
            adjust_learning_rate_D(cnn_optimizer, args.lr, iteration, args.iteration)

            ######################
            # validate the model #
            ######################
            if iteration % args.inter == 0:
                cnn.eval()
                class_correct = list(0.0 for i in range(2))
                class_total = list(0.0 for i in range(2))
                pred_all = []
                real_all = []
                person_all = []
                for data, target, person in test_dataloader:
                    data = data.cuda()
                    target = target.cuda()

                    output = cnn(data)
                    loss = criterion(output, target)
                    _, pred = torch.max(output, 1)

                    pred_all += list(np.array(pred.cpu()))
                    real_all += list(np.array(target.cpu()))
                    person_all += list(np.array(person))
                    correct_tensor = pred.eq(target.data.view_as(pred))

                    correct = (
                        np.squeeze(correct_tensor.numpy())
                        if not args.cuda
                        else np.squeeze(correct_tensor.cpu().numpy())
                    )

                    # calculate accuracy for each object class
                    for i in range(len(target)):
                        if len(target) == 1:
                            label = target.data[i]
                            class_correct[label] += correct.item()
                            class_total[label] += 1
                        else:
                            label = target.data[i]
                            class_correct[label] += correct[i].item()
                            class_total[label] += 1
                    # update average validation loss
                    valid_loss += loss.item() * data.size(0)
                print(len(pred_all), pred_all)
                print(real_all)
                print(person_all)
                a, b = person_acc(pred_all, real_all, person_all)
                # calculate average losses
                io.cprint(a)
                io.cprint(b)
                for i in range(2):
                    if class_total[i] > 0:
                        io.cprint(
                            "Eva Accuracy of %5s: %2d%% (%2d/%2d)"
                            % (
                                classes[i],
                                100 * class_correct[i] / class_total[i],
                                np.sum(class_correct[i]),
                                np.sum(class_total[i]),
                            )
                        )
                        if i == 0:
                            test_acc_0.append(100 * class_correct[i] / class_total[i])
                        if i == 1:
                            test_acc_1.append(100 * class_correct[i] / class_total[i])

                    else:
                        io.cprint(
                            "Eva Accuracy of %5s: N/A (no training examples)"
                            % (classes[i])
                        )

                io.cprint(
                    "\nEva Accuracy (Overall): %2d%% (%2d/%2d)"
                    % (
                        100.0 * np.sum(class_correct) / np.sum(class_total),
                        np.sum(class_correct),
                        np.sum(class_total),
                    )
                )
                test_acc = 100.0 * np.sum(class_correct) / np.sum(class_total)
                test_acc_y.append(test_acc)

                train_loss = train_loss / len(train_dataloader.dataset)
                valid_loss = valid_loss / len(test_dataloader.dataset)

                # print training/validation statistics
                io.cprint(
                    "Epoch: {} \tTraining Loss: {:.6f} \tEva Loss: {:.6f}".format(
                        iteration, train_loss, valid_loss
                    )
                )

                # you can save model if validation loss has decreased, we save the final model
                # if args.device == "3090":
                #     torch.save(
                #         cnn.state_dict(),
                #         path + "cnn" + model_name,
                #         _use_new_zipfile_serialization=False,
                #     )
                # else:
                #     torch.save(cnn.state_dict(), path + "cnn" + model_name)
                train_loss_.append(train_loss)
                test_loss_y.append(valid_loss)

                io.cprint("=====================================")
                io.cprint("+++++++++++++++++++++++++++++++++++++")

        model_name = str(fold) + ".pkl"
        if args.device == "3090":
            torch.save(
                cnn.state_dict(),
                path + "cnn" + model_name,
                _use_new_zipfile_serialization=False,
            )
        else:
            torch.save(cnn.state_dict(), path + "cnn" + model_name)

        acc_5.append(test_acc_y)
        paint_in_pre_train(
            train_loss_,
            test_loss_y,
            test_acc_0,
            test_acc_1,
            test_acc_y,
            path,
            fold,
            args.inter,
        )
    fold_(acc_5, io)
