from __future__ import print_function

import argparse
import os

# Training settings
def get_args():
    parser = argparse.ArgumentParser(
        description="Graph Neural Networks"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="debug_vx",
        metavar="N",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--weight-decay",
        type=int,
        default=1e-4,
        metavar="N",
        help="weight-decay (default: 1e-6)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        metavar="batch_size",
        help="Size of batch)",
    )
    parser.add_argument(
        "--batch_size_test",
        type=int,
        default=32,
        metavar="batch_size",
        help="Size of batch)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        metavar="N",
        help="number of epochs to train ",
    )
    parser.add_argument(
        "--decay_interval",
        type=int,
        default=10000,
        metavar="N",
        help="Learning rate decay interval",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="enables CUDA training"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=3000,
        metavar="N",
        help="how many batches between each model saving",
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches between each cam_result",
    )
    parser.add_argument(
        "--test_N_way",
        type=int,
        default=2,
        metavar="N",
        help="Number of classes for doing each classification run",
    )
    parser.add_argument(
        "--train_N_way",
        type=int,
        default=2,
        metavar="N",
        help="Number of classes for doing each training comparison",
    )
    parser.add_argument(
        "--test_N_shots",
        type=int,
        default=7,
        metavar="N",
        help="Number of shots in cam_result",
    )
    parser.add_argument(
        "--train_N_shots",
        type=int,
        default=7,
        metavar="N",
        help="Number of shots when training",
    )
    parser.add_argument(
        "--unlabeled_extra",
        type=int,
        default=0,
        metavar="N",
        help="Number of shots when training",
    )
    parser.add_argument(
        "--metric_network",
        type=str,
        default="gnn_iclr_nl",
        metavar="N",
        help="gnn_iclr_nl" + "gnn_iclr_active",
    )
    parser.add_argument(
        "--active_random", type=int, default=0, metavar="N", help="random active ? "
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data//psp_data.pkl",
        metavar="N",
        help="Root dataset",
    )
    parser.add_argument(
        "--test_samples", type=int, default=300, metavar="N", help="Number of shots, this parameter has been dropped"
    )
    parser.add_argument(
        "--dataset", type=str, default="mini_imagenet", metavar="N", help="omniglot"
    )
    parser.add_argument(
        "--dec_lr",
        type=int,
        default=10000,
        metavar="N",
        help="Decreasing the learning rate every x iterations",
    )
    parser.add_argument("--data_train", type=int, default=1, metavar="N")
    parser.add_argument("--network", type=int, default=1, metavar="N")
    parser.add_argument("--loss_metric", type=int, default=1, metavar="N")
    parser.add_argument("--enc_nn_train", type=int, default=1, metavar="N")
    parser.add_argument("--att_type", type=str, default="CBAM")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        metavar="N",
        help="input visible devices for training (default: 0)",
    )
    parser.add_argument(
        "--model", type=str, default="ResNetCBAM", help="model (ResNet, ResNetCBAM)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        metavar="str",
        help="Optimizer (default: Adam)",
    )
    parser.add_argument(
        "--r", type=int, default=340, metavar="seed", help="random seed"
    )
    return parser.parse_args()


args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import matplotlib

matplotlib.use("AGG")
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data.data import Train_Generator
from models.GNN import MetricNN, create_models, SoftmaxModule
from models.loss import w_loss
from models.mynet_dirk_full_mask import resnet18_dirk_full_mask
from models.mynet_mask import resnet18_mask
from test_utils import test_one_shot
from utils import io_utils
from utils.util import (
    get_time,
    paint_in_gnn_train,
    adjust_learning_rate,
    fold_,
    setup_seed,
    new_path,
    save_arg,
    adjust_learning_rate_D_List,
)


# cur_time = get_time()
cur_time = "r{}".format(args.r)

data_train_name = [
    "train_{}.pkl",
    "amp_balance_enlarge_train_{}_45_r{}.pkl",
    "balance_enlarge_train_no_slice_{}.pkl",
    "only_enlarge_train_{}.pkl",
]
data_test_name = [
    "test_{}.pkl",
    "only_enlarge_test_{}_45_r{}.pkl",
    "only_enlarge_test_{}_45_r{}.pkl",
]
args.cuda = not args.no_cuda and torch.cuda.is_available()
setup_seed()


def train_batch(model, data):
    [enc_nn, metric_nn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi, oracles_yi, labels_w] = data

    # Compute embedding from x and xi_s
    z = enc_nn(batch_x)[-1]
    zi_s = [enc_nn(batch_xi)[-1] for batch_xi in batches_xi]

    # Compute metric from embeddings
    w_all, out_metric, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi])
    log_soft_prob = softmax_module.forward(out_logits)

    # Loss
    label_x_numpy = label_x.cpu().data.numpy()
    formatted_label_x = np.argmax(label_x_numpy, axis=1)
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))

    formatted_label_x = formatted_label_x.cuda()
    labels_w_v = labels_w.cuda()

    mask_L1 = torch.norm(enc_nn.mask, p=1) / 256
    loss = F.nll_loss(log_soft_prob, formatted_label_x) + mask_L1
    if args.loss_metric == 1:
        my_loss_ = w_loss()
        loss_w_new = my_loss_(w_all, labels_w_v, loss)
        loss_w_new.backward()
        return loss_w_new
    if args.loss_metric == 0:
        loss.backward()
    return loss


def train():
    acc_5_1 = []
    f1_1 = []
    acc_5_2 = []
    f1_2 = []
    acc_5_3 = []
    f1_3 = []
    for fold in range(5):
        name = cur_time + "-{}-mask-shot-{}-bt-{}-lr{}-fold-{}-45".format(
            args.model, args.test_N_shots, args.batch_size, args.lr, fold
        )
        save_path = new_path("checkpoints/{}".format(name))
        io = io_utils.IOStream(save_path + "/run.log")
        io1 = io_utils.IOStream(save_path + "/run_1.log")
        io2 = io_utils.IOStream(save_path + "/run_2.log")
        io3 = io_utils.IOStream(save_path + "/run_3.log")
        save_arg(args, save_path + "/run.log")

        all_data_path = "../processed_data/"
        train_data_path = all_data_path + data_train_name[args.data_train]
        train_loader = Train_Generator((train_data_path).format(str(fold), args.r))
        io.cprint("Batch size: " + str(args.batch_size))

        # Try to load models
        if args.network == 1:

            if args.model == "ResNet":
                enc_nn = resnet18_mask()
                pre_trianed_dict = torch.load(
                    "checkpoints/ResNet_bt_64_iter_80000_testamp_0_pre_train_balance_"
                    "r{}/cnn{}.pkl".format(args.r, fold)
                )
            elif args.model == "ResNetCBAM":
                enc_nn = resnet18_dirk_full_mask()
                pre_trianed_dict = torch.load(
                    "checkpoints/ResNetCBAM_bt_64_iter_80000_testamp_0_pre_train_balance_45_"
                    "r{}/cnn{}.pkl".format(args.r, fold)
                )

            elif args.model == "ResNet-Val":
                enc_nn = resnet18_mask()
                pre_trianed_dict = torch.load(
                    "checkpoints/ResNet_bt_24_iter_80000_testamp_0_pre_train_balance_2022_01_28_00_34"
                    "/cnn{}.pkl".format(fold)
                )
            elif args.model == "ResNetCBAM-Val":
                enc_nn = resnet18_dirk_full_mask()
                pre_trianed_dict = torch.load(
                    "checkpoints/ResNetCBAM_bt_64_iter_80000_testamp_0_pre_train_balance_"
                    "2022_01_28_00_57/cnn{}.pkl".format(fold)
                )

            model_dict = enc_nn.state_dict()
            pre_trianed_dict = {
                k: v for k, v in pre_trianed_dict.items() if k in model_dict
            }
            model_dict.update(pre_trianed_dict)
            enc_nn.load_state_dict(model_dict)
            print("load model!!!")

        elif args.network == 0:
            enc_nn, metric_nn = create_models(args=args)

        metric_nn = MetricNN(args, emb_size=256)
        softmax_module = SoftmaxModule()

        enc_nn.cuda()
        metric_nn.cuda()

        io.cprint(str(enc_nn))
        io.cprint(str(metric_nn))
        opt_enc_nn = optim.Adam(
            enc_nn.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        opt_metric_nn = optim.Adam(
            metric_nn.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        if args.optimizer == "SGD":
            opt_enc_nn = optim.SGD(
                enc_nn.parameters(),
                lr=args.lr,
                momentum=0.99,
                weight_decay=args.weight_decay,
            )
            opt_metric_nn = optim.SGD(
                metric_nn.parameters(),
                lr=args.lr,
                momentum=0.99,
                weight_decay=args.weight_decay,
            )

        if args.enc_nn_train == 1:
            enc_nn.train()

        metric_nn.train()
        counter = 0
        total_loss = 0
        val_acc, val_acc_aux = 0, 0
        test_acc = 0
        train_loss = []
        test_acc_y_3 = []
        test_f1_y_3 = []
        test_loss_y_3 = []
        for batch_idx in range(args.iterations):

            ####################
            # Train
            ####################
            data = train_loader.get_task_batch(
                batch_size=args.batch_size,
                n_way=args.train_N_way,
                num_shots=args.train_N_shots,
                unlabeled_extra=0,
                cuda=args.cuda,
                variable=False,
            )
            [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi, labels_w] = data
            if args.enc_nn_train == 1:
                opt_enc_nn.zero_grad()
            opt_metric_nn.zero_grad()

            loss_d_metric = train_batch(
                model=[enc_nn, metric_nn, softmax_module],
                data=[batch_x, label_x, batches_xi, labels_yi, oracles_yi, labels_w],
            )
            if args.enc_nn_train == 1:
                opt_enc_nn.step()
            opt_metric_nn.step()
            # Parameter containing
            io.cprint(str(metric_nn.gnn_obj.layer_last.fc.weight))
            adjust_learning_rate(
                optimizers=[opt_enc_nn, opt_metric_nn],
                lr=args.lr,
                iter=batch_idx,
                args=args,
            )

            ####################
            # Display
            ####################
            counter += 1
            total_loss += loss_d_metric.item()
            if batch_idx % args.log_interval == 0:
                display_str = "Train Iter: {}".format(batch_idx)
                display_str += "\tLoss_d_metric: {:.6f}".format(total_loss / counter)
                train_loss.append(total_loss / counter)
                io.cprint(display_str)
                io.cprint(str(enc_nn.state_dict()["mask"]))
                counter = 0
                total_loss = 0

            ####################
            # Eva
            ####################
            if (batch_idx + 1) % args.test_interval == 0 and batch_idx >= 20:

                if batch_idx == 20:
                    test_samples = 100
                else:
                    test_samples = 300

                test_data_path = all_data_path + data_test_name[2]
                # you can save model if validation loss has decreased, we calculate results after all training iterations
                with torch.no_grad():
                    test_acc_aux_3, test_loss_3, test_f1_aux_3 = test_one_shot(
                        args,
                        io,
                        model=[enc_nn, metric_nn, softmax_module],
                        train_data_path=train_data_path.format(str(fold), args.r),
                        test_data_path=test_data_path.format(str(fold), args.r),
                        test_samples=test_samples * 5,
                        partition="cam_result",
                    )
                if args.enc_nn_train == 1:
                    enc_nn.train()
                metric_nn.train()

                test_acc_y_3.append(test_acc_aux_3)
                test_f1_y_3.append(test_f1_aux_3)
                test_loss_y_3.append(test_loss_3)
                if test_acc_aux_3 is not None and test_acc_aux_3 >= test_acc:
                    test_acc = test_acc_aux_3
                if args.dataset == "mini_imagenet":
                    io.cprint("Best result accuracy {:.4f} \n".format(test_acc))

        acc_5_3.append(test_acc_y_3)
        f1_3.append(test_f1_y_3)
        paint_in_gnn_train(
            train_loss,
            test_loss_y_3,
            test_acc_y_3,
            test_f1_y_3,
            name,
            io3,
            test_num="3",
        )

    fold_(acc_5_3, io3, f1="acc")
    fold_(f1_3, io3, f1="f1")


if __name__ == "__main__":
    train()
