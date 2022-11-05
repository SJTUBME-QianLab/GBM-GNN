import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable

from data.data import TestGenerator, TestGenerator_p, TestGenerator_p_random


def ROC(y, y_pre):
    y_pre = np.array(y_pre)
    y_score = np.exp(y_pre)
    # print((y_score))
    y_test = label_binarize(y, classes=[0, 1])

    fpr, tpr, thresholds_ = roc_curve(y_test, y_score[:, 1], drop_intermediate=True)
    # print(thresholds_)
    auc_ = auc(fpr, tpr)
    return fpr, tpr, auc_


def Spe(TN, FP):
    return TN / (FP + TN)


def Sen(TP, FN):
    return TP / (TP + FN)


class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, w_all, labels_w, loss):  # 定义前向的函数运算即可
        w_loss = 0
        for i in range(len(w_all)):
            # print(type(labels_w[i]))
            # print(type(w_all[i]))
            # print((labels_w).size())
            # print((w_all[i]).size())
            w_loss += torch.norm((labels_w) - (w_all[i]), p=1)

        return loss + 0.001 * w_loss


def test_one_shot(
    args,
    io,
    model,
    train_data_path="",
    test_data_path="",
    test_samples=5000,
    partition="cam_result",
):
    io = io

    io.cprint("\n**** TESTING WITH %s ***" % (partition,))

    loader = TestGenerator(train_data_path, test_data_path)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    pre_all = []
    pre_all_num = []
    real_all = []

    slice_num, patient_slice_0, patient_slice_1 = loader.get_test_num()

    # 按照病人来预测
    pre_patient = []
    real_patient = []
    # 标签为0的病人
    for patient in range(len(patient_slice_0)):
        real_patient.append(0)
        pre_slice = []
        # 在一个batch内对单张图片进行预测，在batch内投票
        for s in range(patient_slice_0[patient]):
            data = loader.get_task_batch(
                0,
                patient,
                s,
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                unlabeled_extra=args.unlabeled_extra,
                cuda=args.cuda,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                # hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi_cpu

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            # hidden_labels = Variable(hidden_labels)
            x = Variable(x)

            # Compute embedding from x and xi_s
            # z = enc_nn(x)
            # zi_s = [enc_nn(batch_xi) for batch_xi in xi_s]
            z = enc_nn(x)[-1]
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

            # Compute metric from embeddings
            w_all, output, out_logits = metric_nn(
                inputs=[z, zi_s, labels_yi, oracles_yi]
            )
            output = out_logits
            Y = softmax_module.forward(output)
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_inter = [list(y_i) for y_i in y_pred]
            pre_all_num = pre_all_num + list(y_inter)
            y_pred = np.argmax(y_pred, axis=1)

            # slice投票
            if sum(y_pred) <= args.batch_size_test / 2:
                pre_slice.append(0)
            else:
                pre_slice.append(1)

            labels_x_cpu = labels_x_cpu.cpu().numpy()
            labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
            pre_all = pre_all + list(y_pred)
            real_all = real_all + list(labels_x_cpu)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

        if sum(pre_slice) <= patient_slice_0[patient] / 2:
            pre_patient.append(0)
        else:
            pre_patient.append(1)

        # if (i+1) % 100 == 0:
        #     io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    for patient in range(len(patient_slice_1)):  # 标签为1的病人
        real_patient.append(1)
        pre_slice = []
        for s in range(patient_slice_1[patient]):  # 在一个batch内对单张图片进行预测，在batch内投票
            data = loader.get_task_batch(
                1,
                patient,
                s,
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                unlabeled_extra=args.unlabeled_extra,
                cuda=args.cuda,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                # hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi_cpu

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            # hidden_labels = Variable(hidden_labels)
            x = Variable(x)

            # Compute embedding from x and xi_s
            # z = enc_nn(x)
            # zi_s = [enc_nn(batch_xi) for batch_xi in xi_s]
            z = enc_nn(x)[-1]
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

            # Compute metric from embeddings
            w_all, output, out_logits = metric_nn(
                inputs=[z, zi_s, labels_yi, oracles_yi]
            )
            output = out_logits
            Y = softmax_module.forward(output)
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_inter = [list(y_i) for y_i in y_pred]
            pre_all_num = pre_all_num + list(y_inter)
            y_pred = np.argmax(y_pred, axis=1)

            # slice投票
            if sum(y_pred) <= args.batch_size_test / 2:
                pre_slice.append(0)
            else:
                pre_slice.append(1)

            labels_x_cpu = labels_x_cpu.cpu().numpy()
            labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
            pre_all = pre_all + list(y_pred)
            real_all = real_all + list(labels_x_cpu)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

        if sum(pre_slice) <= patient_slice_1[patient] / 2:
            pre_patient.append(0)
        else:
            pre_patient.append(1)

        # if (i+1) % 100 == 0:
        #     io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    labels_x_cpu = Variable(torch.cuda.LongTensor(labels_x_cpu))
    labels_w_v = Variable((labels_w))
    loss_test = F.nll_loss(Y, labels_x_cpu)
    if args.loss_metric == 1:
        my_loss_ = My_loss()
        # print('labels_w_v', labels_w_v.size())
        loss_w_new = my_loss_(w_all, labels_w_v, loss_test)
        loss_test_f = float(loss_w_new)
    if args.loss_metric == 0:
        loss_test_f = float(loss_test)
    del loss_test
    juzhen_all = np.zeros((2, 2))

    # 按照slice得到的数值
    for n_s in range(len(real_all)):
        for num in range(2):
            for num_label in range(2):
                if pre_all[n_s] == num and real_all[n_s] == num_label:
                    juzhen_all[num_label, num] = juzhen_all[num_label, num] + 1

    spe = Spe(juzhen_all[0][0], juzhen_all[0][1])
    sen = Sen(juzhen_all[1][1], juzhen_all[1][0])
    ### 这个不是真正的f1真正的f1需要自己再算一下。
    f1 = 2 * spe * sen / (spe + sen)
    io.ccprint("###slice####")
    io.ccprint("juzhen_all xianshi :")
    io.ccprint(str(juzhen_all))
    io.ccprint("real_label:  " + str(real_all))
    io.ccprint("pre_all:  " + str(pre_all))
    io.ccprint("pre_all_num:  " + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.3f}%)".format(
            correct, total, 100.0 * correct / total
        )
    )
    io.cprint("sen:" + str(sen) + " spe:" + str(spe) + " f1:" + str(f1))

    # 按照patient得到的数值
    juzhen_patient = np.zeros((2, 2))
    for n_s in range(len(real_patient)):
        for num in range(2):
            for num_label in range(2):
                if pre_patient[n_s] == num and real_patient[n_s] == num_label:
                    juzhen_patient[num_label, num] = juzhen_patient[num_label, num] + 1
    spe = Spe(juzhen_patient[0][0], juzhen_patient[0][1])
    sen = Sen(juzhen_patient[1][1], juzhen_patient[1][0])
    acc = (juzhen_patient[0][0] + juzhen_patient[1][1]) / np.sum(juzhen_patient)
    f1 = 2 * spe * sen / (spe + sen)
    io.ccprint("###patient####")
    io.ccprint("juzhen_patient xianshi :")
    io.ccprint(str(juzhen_patient))
    io.ccprint("real_label:  " + str(real_patient))
    io.ccprint("pre_all:  " + str(pre_patient))
    io.ccprint("pre_all_num:  " + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.3f}%)".format(
            juzhen_patient[0][0] + juzhen_patient[1][1],
            np.sum(juzhen_patient),
            100.0 * acc,
        )
    )

    fpr1, tpr1, auc = ROC(real_all, pre_all_num)
    io.cprint(
        "sen:" + str(sen) + " spe:" + str(spe) + " f1:" + str(f1) + "AUC:" + str(auc)
    )

    io.cprint("*** TEST FINISHED ***\n")
    enc_nn.train()
    metric_nn.train()

    return 100.0 * acc, loss_test_f, f1


def test_one_shot_p(
    args,
    io,
    model,
    p=1,
    train_data_path="",
    test_data_path="",
    test_samples=5000,
    partition="cam_result",
):
    io = io

    io.cprint("\n**** TESTING WITH %s ***" % (partition,))

    loader = TestGenerator_p(train_data_path, test_data_path, p=p)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    pre_all = []
    pre_all_num = []
    real_all = []

    slice_num, patient_slice_0, patient_slice_1 = loader.get_test_num()

    # 按照病人来预测
    pre_patient = []
    real_patient = []
    # 标签为0的病人
    for patient in range(len(patient_slice_0)):
        real_patient.append(0)
        pre_slice = []
        # 在一个batch内对单张图片进行预测，在batch内投票
        for s in range(patient_slice_0[patient]):
            data = loader.get_task_batch(
                0,
                patient,
                s,
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                unlabeled_extra=args.unlabeled_extra,
                cuda=args.cuda,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                # hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi_cpu

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            # hidden_labels = Variable(hidden_labels)
            x = Variable(x)

            # Compute embedding from x and xi_s
            # z = enc_nn(x)
            # zi_s = [enc_nn(batch_xi) for batch_xi in xi_s]
            z = enc_nn(x)[-1]
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

            # Compute metric from embeddings
            w_all, output, out_logits = metric_nn(
                inputs=[z, zi_s, labels_yi, oracles_yi]
            )
            output = out_logits
            Y = softmax_module.forward(output)
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_inter = [list(y_i) for y_i in y_pred]
            pre_all_num = pre_all_num + list(y_inter)
            y_pred = np.argmax(y_pred, axis=1)

            # slice投票
            if sum(y_pred) <= args.batch_size_test / 2:
                pre_slice.append(0)
            else:
                pre_slice.append(1)

            labels_x_cpu = labels_x_cpu.cpu().numpy()
            labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
            pre_all = pre_all + list(y_pred)
            real_all = real_all + list(labels_x_cpu)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

        if sum(pre_slice) <= patient_slice_0[patient] / 2:
            pre_patient.append(0)
        else:
            pre_patient.append(1)

        # if (i+1) % 100 == 0:
        #     io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    for patient in range(len(patient_slice_1)):  # 标签为1的病人
        real_patient.append(1)
        pre_slice = []
        for s in range(patient_slice_1[patient]):  # 在一个batch内对单张图片进行预测，在batch内投票
            data = loader.get_task_batch(
                1,
                patient,
                s,
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                unlabeled_extra=args.unlabeled_extra,
                cuda=args.cuda,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                # hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi_cpu

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            # hidden_labels = Variable(hidden_labels)
            x = Variable(x)

            # Compute embedding from x and xi_s
            # z = enc_nn(x)
            # zi_s = [enc_nn(batch_xi) for batch_xi in xi_s]
            z = enc_nn(x)[-1]
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

            # Compute metric from embeddings
            w_all, output, out_logits = metric_nn(
                inputs=[z, zi_s, labels_yi, oracles_yi]
            )
            output = out_logits
            Y = softmax_module.forward(output)
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_inter = [list(y_i) for y_i in y_pred]
            pre_all_num = pre_all_num + list(y_inter)
            y_pred = np.argmax(y_pred, axis=1)

            # slice投票
            if sum(y_pred) <= args.batch_size_test / 2:
                pre_slice.append(0)
            else:
                pre_slice.append(1)

            labels_x_cpu = labels_x_cpu.cpu().numpy()
            labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
            pre_all = pre_all + list(y_pred)
            real_all = real_all + list(labels_x_cpu)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

        if sum(pre_slice) <= patient_slice_1[patient] / 2:
            pre_patient.append(0)
        else:
            pre_patient.append(1)

        # if (i+1) % 100 == 0:
        #     io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    labels_x_cpu = Variable(torch.cuda.LongTensor(labels_x_cpu))
    labels_w_v = Variable((labels_w))
    loss_test = F.nll_loss(Y, labels_x_cpu)
    if args.loss_metric == 1:
        my_loss_ = My_loss()
        # print('labels_w_v', labels_w_v.size())
        loss_w_new = my_loss_(w_all, labels_w_v, loss_test)
        loss_test_f = float(loss_w_new)
    if args.loss_metric == 0:
        loss_test_f = float(loss_test)
    del loss_test
    juzhen_all = np.zeros((2, 2))

    # 按照slice得到的数值
    for n_s in range(len(real_all)):
        for num in range(2):
            for num_label in range(2):
                if pre_all[n_s] == num and real_all[n_s] == num_label:
                    juzhen_all[num_label, num] = juzhen_all[num_label, num] + 1

    spe = Spe(juzhen_all[0][0], juzhen_all[0][1])
    sen = Sen(juzhen_all[1][1], juzhen_all[1][0])
    ### 这个不是真正的f1真正的f1需要自己再算一下。
    f1 = 2 * spe * sen / (spe + sen)
    io.ccprint("###slice####")
    io.ccprint("juzhen_all xianshi :")
    io.ccprint(str(juzhen_all))
    io.ccprint("real_label:  " + str(real_all))
    io.ccprint("pre_all:  " + str(pre_all))
    io.ccprint("pre_all_num:  " + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.3f}%)".format(
            correct, total, 100.0 * correct / total
        )
    )
    io.cprint("sen:" + str(sen) + " spe:" + str(spe) + " f1:" + str(f1))

    # 按照patient得到的数值
    juzhen_patient = np.zeros((2, 2))
    for n_s in range(len(real_patient)):
        for num in range(2):
            for num_label in range(2):
                if pre_patient[n_s] == num and real_patient[n_s] == num_label:
                    juzhen_patient[num_label, num] = juzhen_patient[num_label, num] + 1
    spe = Spe(juzhen_patient[0][0], juzhen_patient[0][1])
    sen = Sen(juzhen_patient[1][1], juzhen_patient[1][0])
    acc = (juzhen_patient[0][0] + juzhen_patient[1][1]) / np.sum(juzhen_patient)
    f1 = 2 * spe * sen / (spe + sen)
    io.ccprint("###patient####")
    io.ccprint("juzhen_patient xianshi :")
    io.ccprint(str(juzhen_patient))
    io.ccprint("real_label:  " + str(real_patient))
    io.ccprint("pre_all:  " + str(pre_patient))
    io.ccprint("pre_all_num:  " + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.3f}%)".format(
            juzhen_patient[0][0] + juzhen_patient[1][1],
            np.sum(juzhen_patient),
            100.0 * acc,
        )
    )

    fpr1, tpr1, auc = ROC(real_all, pre_all_num)
    io.cprint(
        "sen:" + str(sen) + " spe:" + str(spe) + " f1:" + str(f1) + "AUC:" + str(auc)
    )

    io.cprint("*** TEST FINISHED ***\n")
    enc_nn.train()
    metric_nn.train()

    return 100.0 * acc, loss_test_f, f1


def test_one_shot_p_random(
    args,
    io,
    model,
    p=1,
    train_data_path="",
    test_data_path="",
    test_samples=5000,
    partition="cam_result",
):
    io = io

    io.cprint("\n**** TESTING WITH %s ***" % (partition,))

    loader = TestGenerator_p_random(train_data_path, test_data_path, p=p)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    pre_all = []
    pre_all_num = []
    real_all = []

    slice_num, patient_slice_0, patient_slice_1 = loader.get_test_num()

    # 按照病人来预测
    pre_patient = []
    real_patient = []
    # 标签为0的病人
    for patient in range(len(patient_slice_0)):
        real_patient.append(0)
        pre_slice = []
        # 在一个batch内对单张图片进行预测，在batch内投票
        for s in range(patient_slice_0[patient]):
            data = loader.get_task_batch(
                0,
                patient,
                s,
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                unlabeled_extra=args.unlabeled_extra,
                cuda=args.cuda,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                # hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi_cpu

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            # hidden_labels = Variable(hidden_labels)
            x = Variable(x)

            # Compute embedding from x and xi_s
            # z = enc_nn(x)
            # zi_s = [enc_nn(batch_xi) for batch_xi in xi_s]
            z = enc_nn(x)[-1]
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

            # Compute metric from embeddings
            w_all, output, out_logits = metric_nn(
                inputs=[z, zi_s, labels_yi, oracles_yi]
            )
            output = out_logits
            Y = softmax_module.forward(output)
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_inter = [list(y_i) for y_i in y_pred]
            pre_all_num = pre_all_num + list(y_inter)
            y_pred = np.argmax(y_pred, axis=1)

            # slice投票
            if sum(y_pred) <= args.batch_size_test / 2:
                pre_slice.append(0)
            else:
                pre_slice.append(1)

            labels_x_cpu = labels_x_cpu.cpu().numpy()
            labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
            pre_all = pre_all + list(y_pred)
            real_all = real_all + list(labels_x_cpu)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

        if sum(pre_slice) <= patient_slice_0[patient] / 2:
            pre_patient.append(0)
        else:
            pre_patient.append(1)

        # if (i+1) % 100 == 0:
        #     io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    for patient in range(len(patient_slice_1)):  # 标签为1的病人
        real_patient.append(1)
        pre_slice = []
        for s in range(patient_slice_1[patient]):  # 在一个batch内对单张图片进行预测，在batch内投票
            data = loader.get_task_batch(
                1,
                patient,
                s,
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                unlabeled_extra=args.unlabeled_extra,
                cuda=args.cuda,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data

            if args.cuda:
                xi_s = [batch_xi.cuda() for batch_xi in xi_s]
                labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
                oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
                # hidden_labels = hidden_labels.cuda()
                x = x.cuda()
            else:
                labels_yi = labels_yi_cpu

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            # hidden_labels = Variable(hidden_labels)
            x = Variable(x)

            # Compute embedding from x and xi_s
            # z = enc_nn(x)
            # zi_s = [enc_nn(batch_xi) for batch_xi in xi_s]
            z = enc_nn(x)[-1]
            zi_s = [enc_nn(batch_xi)[-1] for batch_xi in xi_s]

            # Compute metric from embeddings
            w_all, output, out_logits = metric_nn(
                inputs=[z, zi_s, labels_yi, oracles_yi]
            )
            output = out_logits
            Y = softmax_module.forward(output)
            y_pred = softmax_module.forward(output)
            y_pred = y_pred.data.cpu().numpy()
            y_inter = [list(y_i) for y_i in y_pred]
            pre_all_num = pre_all_num + list(y_inter)
            y_pred = np.argmax(y_pred, axis=1)

            # slice投票
            if sum(y_pred) <= args.batch_size_test / 2:
                pre_slice.append(0)
            else:
                pre_slice.append(1)

            labels_x_cpu = labels_x_cpu.cpu().numpy()
            labels_x_cpu = np.argmax(labels_x_cpu, axis=1)
            pre_all = pre_all + list(y_pred)
            real_all = real_all + list(labels_x_cpu)

            for row_i in range(y_pred.shape[0]):
                if y_pred[row_i] == labels_x_cpu[row_i]:
                    correct += 1
                total += 1

        if sum(pre_slice) <= patient_slice_1[patient] / 2:
            pre_patient.append(0)
        else:
            pre_patient.append(1)

        # if (i+1) % 100 == 0:
        #     io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))

    labels_x_cpu = Variable(torch.cuda.LongTensor(labels_x_cpu))
    labels_w_v = Variable((labels_w))
    loss_test = F.nll_loss(Y, labels_x_cpu)
    if args.loss_metric == 1:
        my_loss_ = My_loss()
        # print('labels_w_v', labels_w_v.size())
        loss_w_new = my_loss_(w_all, labels_w_v, loss_test)
        loss_test_f = float(loss_w_new)
    if args.loss_metric == 0:
        loss_test_f = float(loss_test)
    del loss_test
    juzhen_all = np.zeros((2, 2))

    # 按照slice得到的数值
    for n_s in range(len(real_all)):
        for num in range(2):
            for num_label in range(2):
                if pre_all[n_s] == num and real_all[n_s] == num_label:
                    juzhen_all[num_label, num] = juzhen_all[num_label, num] + 1

    spe = Spe(juzhen_all[0][0], juzhen_all[0][1])
    sen = Sen(juzhen_all[1][1], juzhen_all[1][0])
    ### 这个不是真正的f1真正的f1需要自己再算一下。
    f1 = 2 * spe * sen / (spe + sen)
    io.ccprint("###slice####")
    io.ccprint("juzhen_all xianshi :")
    io.ccprint(str(juzhen_all))
    io.ccprint("real_label:  " + str(real_all))
    io.ccprint("pre_all:  " + str(pre_all))
    io.ccprint("pre_all_num:  " + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.3f}%)".format(
            correct, total, 100.0 * correct / total
        )
    )
    io.cprint("sen:" + str(sen) + " spe:" + str(spe) + " f1:" + str(f1))

    # 按照patient得到的数值
    juzhen_patient = np.zeros((2, 2))
    for n_s in range(len(real_patient)):
        for num in range(2):
            for num_label in range(2):
                if pre_patient[n_s] == num and real_patient[n_s] == num_label:
                    juzhen_patient[num_label, num] = juzhen_patient[num_label, num] + 1
    spe = Spe(juzhen_patient[0][0], juzhen_patient[0][1])
    sen = Sen(juzhen_patient[1][1], juzhen_patient[1][0])
    acc = (juzhen_patient[0][0] + juzhen_patient[1][1]) / np.sum(juzhen_patient)
    f1 = 2 * spe * sen / (spe + sen)
    io.ccprint("###patient####")
    io.ccprint("juzhen_patient xianshi :")
    io.ccprint(str(juzhen_patient))
    io.ccprint("real_label:  " + str(real_patient))
    io.ccprint("pre_all:  " + str(pre_patient))
    io.ccprint("pre_all_num:  " + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.3f}%)".format(
            juzhen_patient[0][0] + juzhen_patient[1][1],
            np.sum(juzhen_patient),
            100.0 * acc,
        )
    )

    fpr1, tpr1, auc = ROC(real_all, pre_all_num)
    io.cprint(
        "sen:" + str(sen) + " spe:" + str(spe) + " f1:" + str(f1) + "AUC:" + str(auc)
    )

    io.cprint("*** TEST FINISHED ***\n")
    enc_nn.train()
    metric_nn.train()

    return 100.0 * acc, loss_test_f, f1
