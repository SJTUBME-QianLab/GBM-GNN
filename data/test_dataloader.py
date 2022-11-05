import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.test_datagenerator import TestGenerator
from models.loss import My_loss
from utils.util import Spe, Sen, ROC


def vote_pre(slice_p, y_pre4, y2):
    label_p = 0
    section = []
    for i in range(len(slice_p)):
        section.append(label_p)
        label_p = label_p + slice_p[i]
    section.append(label_p)
    # print(section)

    pre = []
    # 然后计算里面的多数票
    for i in range(len(slice_p)):
        y_pre = y_pre4[section[i] : section[i + 1]]
        y_pre = np.array(y_pre)
        y_score = np.exp(y_pre)
        index_0 = 0
        index_1 = 0
        for y_pre_num in y_score:
            if list(y_pre_num).index(max(y_pre_num)) == 1:
                index_1 += 1
            if list(y_pre_num).index(max(y_pre_num)) == 0:
                index_0 += 1
        if index_0 >= index_1:
            count = []
            for i in range(len(y_pre)):
                count.append(y_pre[i][0])
            pre.append(y_score[count.index(max(np.array(count)))])
        if index_0 < index_1:
            count = []
            for i in range(len(y_pre)):
                count.append(y_pre[i][1])
            pre.append(y_score[count.index(max(np.array(count)))])

    return pre, y2


def test_one_shot(
    args, io, model, train_data_path="", test_data_path="", partition="test"
):
    io = io

    io.cprint("\n**** TESTING WITH %s ***" % (partition,))
    loader = TestGenerator(train_data_path, test_data_path)
    [mask_data, metric_nn, softmax_module] = model
    metric_nn.eval()
    mask_data.eval()
    correct = 0
    total = 0
    pre_all = []
    pre_all_num = []
    real_all = []

    slice_num, patient_slice_0, patient_slice_1 = loader.get_test_num()

    # 按照病人来预测
    pre_patient = []
    real_patient = []
    for patient in range(len(patient_slice_0)):  # 标签为0的病人
        real_patient.append(0)
        pre_slice = []
        for s in range(patient_slice_0[patient]):  # 在一个batch内对单张图片进行预测，在batch内投票
            data = loader.get_task_batch(
                0,
                patient,
                s,
                batch_size=args.batch_size_test,
                n_way=args.test_N_way,
                num_shots=args.test_N_shots,
                unlabeled_extra=args.unlabeled_extra,
                cuda=True,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data
            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            x = x.cuda()

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            x = Variable(x)

            # Compute embedding from x and xi_s
            _, z = mask_data(x)
            zi_s = [mask_data(batch_xi)[1] for batch_xi in xi_s]

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
                cuda=True,
                variable=False,
            )
            [x, labels_x_cpu, _, _, xi_s, labels_yi_cpu, oracles_yi, labels_w] = data

            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            oracles_yi = [oracle_yi.cuda() for oracle_yi in oracles_yi]
            # hidden_labels = hidden_labels.cuda()
            x = x.cuda()

            xi_s = [Variable(batch_xi) for batch_xi in xi_s]
            labels_yi = [Variable(label_yi) for label_yi in labels_yi]
            oracles_yi = [Variable(oracle_yi) for oracle_yi in oracles_yi]
            x = Variable(x)

            # Compute embedding from x and xi_s
            _, z = mask_data(x)
            zi_s = [mask_data(batch_xi)[1] for batch_xi in xi_s]

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

    labels_x_cpu = Variable(torch.cuda.LongTensor(labels_x_cpu))
    labels_w_v = Variable((labels_w))
    loss_test = F.nll_loss(Y, labels_x_cpu)
    my_loss_ = My_loss()
    loss_w_new = my_loss_(w_all, labels_w_v, loss_test)
    loss_test_f = float(loss_w_new)
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
    io.ccprint("###slice####")
    io.ccprint("juzhen_all xianshi :")
    io.ccprint(str(juzhen_all))
    # io.ccprint('real_label:  ' + str(real_all))
    # io.ccprint('pre_all:  ' + str(pre_all))
    # io.ccprint('pre_all_num:  ' + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.4f}%)".format(
            correct, total, 100.0 * correct / total
        )
    )
    io.cprint("sen:" + str(sen) + " spe:" + str(spe))

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
    io.ccprint("###patient####")
    io.ccprint("juzhen_patient xianshi :")
    io.ccprint(str(juzhen_patient))
    # io.ccprint('real_label:  ' + str(real_patient))
    # io.ccprint('pre_all:  ' + str(pre_patient))
    # io.ccprint('pre_all_num:  ' + str(pre_all_num))
    io.cprint(
        "{} correct from {} \tAccuracy: {:.4f}%)".format(
            juzhen_patient[0][0] + juzhen_patient[1][1],
            np.sum(juzhen_patient),
            100.0 * acc,
        )
    )

    fpr1, tpr1, auc = ROC(real_all, pre_all_num)

    slice_ = (
        [args.batch_size_test] * 600
        if partition == "TEST"
        else [args.batch_size_test] * 150
    )
    pre, y_p = vote_pre(slice_, pre_all_num, real_patient)
    mean_fpr_bl, mean_tpr_bl, mean_auc_bl = ROC(y_p, pre)
    print(mean_auc_bl)
    io.cprint(
        "sen:"
        + str(sen)
        + " spe:"
        + str(spe)
        + "AUC:"
        + str(auc)
        + "auc_p:"
        + str(mean_auc_bl)
    )

    io.cprint("*** FINISHED ***\n")

    return 100.0 * acc, loss_test_f, mean_auc_bl
