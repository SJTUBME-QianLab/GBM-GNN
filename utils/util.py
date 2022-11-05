#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 19:59
# @Author  : Dirk Li
# @Email   : junli.dirk@gmail.com
# @File    : utils.py
# @Software: PyCharm,Windows10
# @Hardware: 32G-RAM,Intel-i7-7700k,GTX-1080

import os
import pickle
import random
import time

import numpy as np
import torch
import yaml
import matplotlib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

matplotlib.use("AGG")
from matplotlib import pyplot as plt

import cv2


def paint(
    train_loss, test_loss_y, test_acc_0, test_acc_1, test_acc_y, path, fold, inter
):
    path += "/"
    if not os.path.isdir(path):
        os.makedirs(path)
    print("train_loss", train_loss)
    print("test_loss_y", test_loss_y)
    print("test_acc_y", test_acc_y)

    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(train_loss)), train_loss, label="train_loss")

    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss")
    plt.title("train")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率

    plt.savefig("{}{}_fold_{}.png".format(path, "train", fold), dpi=300)
    # cam_result
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_loss_y)), test_loss_y, label="test_loss")

    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率

    plt.savefig("{}{}_fold_{}.png".format(path, "cam_result", fold), dpi=300)

    # test_acc
    plt.figure(figsize=(12, 4))

    plt.plot(range(0, len(test_acc_y)), test_acc_y, label="test_acc")
    plt.plot(range(0, len(test_acc_0)), test_acc_0, label="test_acc_0")
    plt.plot(range(0, len(test_acc_1)), test_acc_1, label="test_acc_1")
    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss_acc")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率

    plt.savefig("{}{}_fold_{}.png".format(path, "test_acc", fold), dpi=300)


def ROC(y, y_pre):
    y_pre = np.array(y_pre)
    y_score = y_pre
    y_test = label_binarize(y, classes=[0, 1])
    fpr, tpr, thresholds_ = roc_curve(y_test, y_score, drop_intermediate=True)
    auc_ = auc(fpr, tpr)
    return fpr, tpr, auc_


def ROC_temp(y, y_pre):
    y_pre = np.array(y_pre)
    y_score = np.exp(y_pre)
    y_test = label_binarize(y, classes=[0, 1])
    fpr, tpr, thresholds_ = roc_curve(y_test, y_score[:, 1], drop_intermediate=True)
    auc_ = auc(fpr, tpr)
    return fpr, tpr, auc_


def Spe(TN, FP):
    return TN / (FP + TN)


def Sen(TP, FN):
    return TP / (TP + FN)


def person_acc(pre, real, person):
    person_0 = 0
    person_1 = 0
    person_0_pre_ture = 0
    person_1_pre_ture = 0
    for p in set(person):
        location = [person[index_] == p for index_ in range(len(person))]
        #         print()
        #         print(np.array(person)[location])
        #         print(np.array(pre)[location])
        #         print(np.array(real)[location])

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
    spe = Spe(person_0_pre_ture, person_0 - person_0_pre_ture)
    sen = Sen(person_1_pre_ture, person_1 - person_1_pre_ture)
    acc = (person_1_pre_ture + person_0_pre_ture) / (person_1 + person_0)

    return (
        ("1:{}//{}".format(person_1_pre_ture, person_1)),
        ("0:{}//{}".format(person_0_pre_ture, person_0)),
        acc,
        spe,
        sen,
    )


def rotate_img(sample, angle, central=True):
    [height, width] = sample.shape
    if central:
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        return cv2.warpAffine(sample, M, (width, height))
    else:
        print("not ready")
        return 0


def flip_img(img, a=1, dst=None):
    return cv2.flip(img, a, dst=dst)


def cut_img(img, size=[350, 300], central=True):
    [height, width] = img.shape
    cut_w = int(size[1] / 2)
    cut_h = int(size[0] / 2)
    img = np.array(img)
    if central:
        return img[
            int(height / 2 - cut_h) : int(height / 2 + cut_h),
            int(width / 2 - cut_w) : int(width / 2 + cut_w),
        ]
    print("not ready now")
    return 0


def rotate_img_nii(nii, angle, central=True):
    image_ = []

    for i in range(nii.shape[0]):
        img = nii[i]
        image_.append(rotate_img(img, angle=angle, central=central))

    return np.array(image_)


def flip_img_nii(nii, a=1, dst=None):
    image_ = []

    for i in range(nii.shape[0]):
        img = nii[i]
        image_.append(flip_img(img, a=a, dst=dst))

    return np.array(image_)


def enlarge_img(sample, size=[218, 182], central=True):
    [height, width] = sample.shape
    sample_new_cv = cv2.resize(
        sample, (int(1.2 * width), int(1.2 * height)), interpolation=cv2.INTER_CUBIC
    )
    return cut_img(sample_new_cv, size=size, central=central)


# 不增加病人，将原图放大，仿射变换，仿射变换 不保留原图，输入4切片病人输出12切片病人
# 只仿射少的那个（1类），对于多的那个类别（0类），只进行放大，由输入参数train_type决定
def amp_data(nii, train_type=0):
    image_ = []

    for i in range(nii.shape[0]):
        # enlarged_img = nii[i]  #不放大了
        enlarged_img = enlarge_img(nii[i])
        # print('enlarge:', enlarged_img.shape)
        image_.append(enlarged_img)
        # if train_type == '1':
        #     image_.append(flip_img(enlarged_img))
        #     image_.append(rotate_img(enlarged_img, -2, central=True))

    return np.array(image_)


# 放大，切片，不增加病人
def amp_dict(dict_path, debug=False):
    with open(dict_path, "rb") as f:
        image = pickle.load(f)
    new_img = {}
    for k in image.keys():
        new_img[k] = []

    for k in image.keys():
        for p in range(len(image[k])):
            if debug:
                print(type(image[k]))
                print("before", np.array(image[k][p]).shape)

            s = np.array(image[k][p]).shape[0]
            if s > 5:
                s = int(s / 2)

                nii = image[k][p][(s - 2) : (s + 3), :, :]
                if debug:
                    print("after", nii.shape)
                # 数据增强
                new_img[k].append(amp_data(nii, train_type=k))

            else:
                nii = image[k][p]
                if debug:
                    print("after", nii.shape)
                new_img[k].append(amp_data(nii, train_type=k))
    if debug:
        for k in image.keys():
            for p in range(len(image[k])):
                print("before", np.array(image[k][p]).shape)
                print("after", np.array(new_img[k][p]).shape)
    return new_img


# 这个函数的作用是把train的少的那个扩增两倍，放大，仿射变换，切片，增加病人
def amp_dict_train_0922(dict_path, debug=False):
    with open(dict_path, "rb") as f:
        image = pickle.load(f)
    new_img = {}
    for k in image.keys():
        new_img[k] = []

    for k in image.keys():
        for p in range(len(image[k])):
            if debug:
                print(type(image[k]))
                print("before", np.array(image[k][p]).shape)

            s = np.array(image[k][p]).shape[0]
            if s > 5:
                s = int(s / 2)

                nii = image[k][p][(s - 2) : (s + 3), :, :]
                nii = amp_data(nii)
                if k == "1":
                    if debug:
                        print("after", nii.shape)
                    # new_img[k].append(affine_tran_img_nii(nii))
                    new_img[k].append(rotate_img_nii(nii, -2, central=True))
                    new_img[k].append(nii)
                    new_img[k].append(flip_img_nii(nii))
                    # new_img[k].append(affine_tran_img_nii(nii, position=[0.05, 0.1, 0.9, 0.05, 0.1, 0.9]))
                if k == "0":
                    new_img[k].append(nii)

            else:
                nii = image[k][p]
                nii = amp_data(nii)
                if debug:
                    print("after", nii.shape)
                # new_img[k].append(amp_data(nii,train_type=k))
                if k == "1":
                    if debug:
                        print("after", nii.shape)
                    new_img[k].append(rotate_img_nii(nii, -2, central=True))
                    new_img[k].append(nii)
                    new_img[k].append(flip_img_nii(nii))
                if k == "0":
                    new_img[k].append(nii)
    if debug:
        for k in image.keys():
            for p in range(len(image[k])):
                print("before", np.array(image[k][p]).shape)
                print("after", np.array(new_img[k][p]).shape)
    return new_img


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_D(optimizer, learning_rate, i_iter, max_iter, power=0.9):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]["lr"] = lr


def adjust_learning_rate_D_List(optimizers, learning_rate, i_iter, max_iter, power=0.9):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    new_lr = lr_poly(learning_rate, i_iter, max_iter, power)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def new_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def get_time():
    cur_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    return cur_time


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_seed2(seed=2022):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_arg(args, path):
    arg_dict = vars(args)
    with open(path, "a") as f:
        yaml.dump(arg_dict, f)


def adjust_learning_rate(optimizers, lr, iter, args):
    new_lr = lr * (0.5 ** (int(iter / args.dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def paint_in_gnn_train(
    train_loss, test_loss_y, test_acc_y, test_f1_y, name, io, test_num="1"
):
    io.cprint("test_acc_y_{} =".format(test_num) + str(test_acc_y))
    io.cprint("test_f1_y_{} = ".format(test_num) + str(test_f1_y))
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(train_loss)), train_loss, label="train_loss")

    plt.xlabel("/20_train_Iter")
    plt.ylabel("loss")
    plt.title("train")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300

    plt.savefig("checkpoints/{}/{}.png".format(name, "train"), dpi=300)
    # cam_result
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_loss_y)), test_loss_y, label="test_loss")

    plt.xlabel("/200_train")
    plt.ylabel("loss")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300

    plt.savefig(
        "checkpoints/{}/{}_{}.png".format(name, test_num, "cam_result"), dpi=300
    )

    # test_acc
    plt.figure(figsize=(12, 4))

    plt.plot(range(0, len(test_acc_y)), test_acc_y, label="test_acc")
    plt.xlabel("/200_train")
    plt.ylabel("loss_acc")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300

    plt.savefig("checkpoints/{}/{}_{}.png".format(name, test_num, "test_acc"), dpi=300)
    # test_f1
    plt.figure(figsize=(12, 4))

    plt.plot(range(0, len(test_f1_y)), test_f1_y, label="test_f1")
    plt.xlabel("/200_train")
    plt.ylabel("f1")
    plt.title("f1_value")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300

    plt.savefig("checkpoints/{}/{}_{}.png".format(name, test_num, "test_f1"), dpi=300)


def paint_in_gnn_train_val_test(
    train_loss,
    val_loss_y,
    val_acc_y,
    val_f1_y,
    test_loss_y,
    test_acc_y,
    test_f1_y,
    name,
    io,
    test_num="1",
):
    io.cprint("test_acc_y_{} =".format(test_num) + str(test_acc_y))
    io.cprint("test_f1_y_{} = ".format(test_num) + str(test_f1_y))
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(train_loss)), train_loss, label="train_loss")
    plt.xlabel("/20_train_Iter")
    plt.ylabel("loss")
    plt.title("train")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("checkpoints/{}/{}.png".format(name, "train"), dpi=300)
    # val
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(val_loss_y)), val_loss_y, label="val_loss")
    plt.xlabel("/200_train")
    plt.ylabel("loss")
    plt.title("val")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("checkpoints/{}/{}_{}.png".format(name, test_num, "val"), dpi=300)
    # cam_result
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_loss_y)), test_loss_y, label="test_loss")
    plt.xlabel("/200_train")
    plt.ylabel("loss")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.savefig(
        "checkpoints/{}/{}_{}.png".format(name, test_num, "cam_result"), dpi=300
    )
    # val_acc
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(val_acc_y)), val_acc_y, label="val_acc")
    plt.xlabel("/200_train")
    plt.ylabel("loss_acc")
    plt.title("val")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("checkpoints/{}/{}_{}.png".format(name, test_num, "val_acc"), dpi=300)
    # test_acc
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_acc_y)), test_acc_y, label="test_acc")
    plt.xlabel("/200_train")
    plt.ylabel("loss_acc")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("checkpoints/{}/{}_{}.png".format(name, test_num, "test_acc"), dpi=300)
    # val_f1
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(val_f1_y)), val_f1_y, label="val_f1")
    plt.xlabel("/200_train")
    plt.ylabel("f1")
    plt.title("f1_value")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("checkpoints/{}/{}_{}.png".format(name, test_num, "val_f1"), dpi=300)
    # test_f1
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_f1_y)), test_f1_y, label="test_f1")
    plt.xlabel("/200_train")
    plt.ylabel("f1")
    plt.title("f1_value")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 300
    plt.savefig("checkpoints/{}/{}_{}.png".format(name, test_num, "test_f1"), dpi=300)


def paint_in_pre_train(
    train_loss, test_loss_y, test_acc_0, test_acc_1, test_acc_y, path, fold, inter
):

    if not os.path.isdir(path):
        os.makedirs(path)
    print("train_loss", train_loss)
    print("test_loss_y", test_loss_y)
    print("test_acc_y", test_acc_y)

    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(train_loss)), train_loss, label="train_loss")

    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss")
    plt.title("train")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率

    plt.savefig("{}{}_fold_{}.png".format(path, "train", fold), dpi=300)
    # cam_result
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_loss_y)), test_loss_y, label="test_loss")

    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率

    plt.savefig("{}{}_fold_{}.png".format(path, "cam_result", fold), dpi=300)

    # test_acc
    plt.figure(figsize=(12, 4))

    plt.plot(range(0, len(test_acc_y)), test_acc_y, label="test_acc")
    plt.plot(range(0, len(test_acc_0)), test_acc_0, label="test_acc_0")
    plt.plot(range(0, len(test_acc_1)), test_acc_1, label="test_acc_1")
    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss_acc")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率

    plt.savefig("{}{}_fold_{}.png".format(path, "test_acc", fold), dpi=300)


def paint_in_pre_train_val_test(
    train_loss,
    val_loss_y,
    val_acc_0,
    val_acc_1,
    val_acc_y,
    test_loss_y,
    test_acc_0,
    test_acc_1,
    test_acc_y,
    path,
    fold,
    inter,
):

    if not os.path.isdir(path):
        os.makedirs(path)
    print("train_loss", train_loss)
    print("val_loss_y", val_loss_y)
    print("val_acc_y", val_acc_y)
    print("test_loss_y", test_loss_y)
    print("test_acc_y", test_acc_y)

    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(train_loss)), train_loss, label="train_loss")

    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss")
    plt.title("train")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率
    plt.savefig("{}{}_fold_{}.png".format(path, "train", fold), dpi=300)

    # cam_result
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_loss_y)), test_loss_y, label="test_loss")
    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率
    plt.savefig("{}{}_fold_{}.png".format(path, "cam_result", fold), dpi=300)

    # val
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(val_loss_y)), val_loss_y, label="val_loss")
    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss")
    plt.title("val")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率
    plt.savefig("{}{}_fold_{}.png".format(path, "val", fold), dpi=300)

    # test_acc
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(test_acc_y)), test_acc_y, label="test_acc")
    plt.plot(range(0, len(test_acc_0)), test_acc_0, label="test_acc_0")
    plt.plot(range(0, len(test_acc_1)), test_acc_1, label="test_acc_1")
    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss_acc")
    plt.title("cam_result")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率
    plt.savefig("{}{}_fold_{}.png".format(path, "test_acc", fold), dpi=300)

    # val_acc
    plt.figure(figsize=(12, 4))
    plt.plot(range(0, len(val_acc_y)), val_acc_y, label="val_acc")
    plt.plot(range(0, len(val_acc_0)), val_acc_0, label="val_acc_0")
    plt.plot(range(0, len(val_acc_1)), val_acc_1, label="val_acc_1")
    plt.xlabel("/{}_batch".format(inter))
    plt.ylabel("loss_acc")
    plt.title("val")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.rcParams["savefig.dpi"] = 300  # 图片像素
    plt.rcParams["figure.dpi"] = 300  # 分辨率
    plt.savefig("{}{}_fold_{}.png".format(path, "val_acc", fold), dpi=300)


def calculate_w(labels_w, n_way, num_shots):
    num = n_way * num_shots + 1
    labels_w = np.array(labels_w)[:, :, 0]
    labels_w = torch.from_numpy(labels_w)

    labels_w = torch.transpose(labels_w, 0, 1)
    labels_w = labels_w.unsqueeze(2)
    labels_w1 = torch.transpose(labels_w, 1, 2)
    # print('size:', labels_w.size())
    # print('size:', labels_w1.size())
    labels_w_new = torch.abs(torch.abs(labels_w - labels_w1) - 1) - torch.eye(num)
    # print('size:', labels_w_new.size())
    # print('+++++_______9')
    # for i in range(21):
    #     print(np.array(labels_w_new[0, i, :]))
    # print('+++++_______9')
    return labels_w_new


def fold_acc_roc_spe_sen(acc_5, acc_5_slice, roc_5, spe_5, sen_5, io):
    acc_5_np = np.array(acc_5)
    average = acc_5_np.mean(axis=0)  # 计算均值
    average_roc = np.array(roc_5).mean(axis=0)
    average_spe = np.array(spe_5).mean(axis=0)
    average_sen = np.array(sen_5).mean(axis=0)
    average_acc_slice = np.array(acc_5_slice).mean(axis=0)

    average = average.tolist()
    average_roc = average_roc.tolist()
    average_spe = average_spe.tolist()
    average_sen = average_sen.tolist()
    average_acc_slice = average_acc_slice.tolist()

    # 每一轮的平均值及每轮的五折结果
    for n in range(len(average)):
        io.cprint("================================")
        io.cprint(str(n) + "====" + str(average[n]))
        for i in range(5):
            io.cprint(
                "ACC:"
                + str(acc_5[i][n])
                + "Slice ACC:"
                + str(acc_5_slice[i][n])
                + "Auc:"
                + str(roc_5[i][n])
                + "spe:"
                + str(spe_5[i][n])
                + "sen:"
                + str(sen_5[i][n])
            )

    # 最高的准确率
    index_h_acc = average.index(max(average))

    io.cprint("index_h_acc" + str(index_h_acc))
    io.cprint("the highest average acc is:" + str(average[index_h_acc]))
    io.cprint("slice acc:" + str(average_acc_slice[index_h_acc]))
    io.cprint("roc:" + str(average_roc[index_h_acc]))
    io.cprint("spe:" + str(average_spe[index_h_acc]))
    io.cprint("sen:" + str(average_sen[index_h_acc]))
    io.cprint("=================================")
    for i in range(5):
        io.cprint(str(acc_5[i][index_h_acc]))


def fold_(acc_5, io, f1="f1"):
    acc_5_np = np.array(acc_5)
    average = acc_5_np.mean(axis=0)
    average = average.tolist()

    for n in range(len(average)):
        io.cprint("=============={}================".format(f1))
        io.cprint(str(n) + "=={}==".format(f1) + str(average[n]))
        for i in range(5):
            io.cprint(str(acc_5[i][n]))

    index_h_acc = average.index(max(average))
    io.cprint("index_h_acc" + str(index_h_acc))
    io.cprint("the highest average {} is:".format(f1) + str(average[index_h_acc]))
    io.cprint("=============={}================".format(f1))
    for i in range(5):
        io.cprint(str(acc_5[i][index_h_acc]))


def fold_val_test(acc_5_val, acc_5_test, io, f1="f1"):
    acc_5_np_val = np.array(acc_5_val)
    average_val = acc_5_np_val.mean(axis=0)
    average_val = average_val.tolist()

    acc_5_np_test = np.array(acc_5_test)
    average_test = acc_5_np_test.mean(axis=0)
    average_test = average_test.tolist()

    for n in range(len(average_val)):
        io.cprint("=============={}================".format(f1))

        io.cprint(str(n) + "====" + str(average_val[n]) + "====" + str(average_test[n]))
        for i in range(5):
            io.cprint("Val_" + str(acc_5_val[i][n]) + "_Test_" + str(acc_5_test[i][n]))

    index_h_acc = average_val.index(max(average_val))
    io.cprint("index_h_acc" + str(index_h_acc))
    io.cprint(
        "the highest average val acc is:"
        + str(average_val[index_h_acc])
        + "the cam_result acc is:"
        + str(average_test[index_h_acc])
    )
    io.cprint("==============================")
    for i in range(5):
        io.cprint(
            "Val_"
            + str(acc_5_val[i][index_h_acc])
            + "_Test_"
            + str(acc_5_test[i][index_h_acc])
        )
