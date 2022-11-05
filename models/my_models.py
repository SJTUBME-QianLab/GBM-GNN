

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# 这个文件用于生成用到的网络模型，一层又一层，主要是模型的选择
# 对每一幅图像做常规卷积操作 得到图像的隐藏状态，也即在另一个空间的表示形式，而已 return [e1, e2, e3, output]

# 使用定义好的网络计算结果
from models.gnn_iclr import GNN_nl, GNN_nl_omniglot, GNN_active


class MetricNN(nn.Module):
    def __init__(self, args, emb_size):
        super(MetricNN, self).__init__()

        self.metric_network = args.metric_network
        self.emb_size = emb_size   # 表示一幅图像的向量维度
        self.args = args

        if self.metric_network == 'gnn_iclr_nl':
            assert(self.args.train_N_way == self.args.test_N_way)
            num_inputs = self.emb_size + self.args.train_N_way
            if self.args.dataset == 'mini_imagenet':
                self.gnn_obj = GNN_nl(args, num_inputs, nf=self.args.hidden_f, J=1)
            elif 'omniglot' in self.args.dataset:
                self.gnn_obj = GNN_nl_omniglot(args, num_inputs, nf=self.args.hidden_f, J=1)
        elif self.metric_network == 'gnn_iclr_active':
            assert(self.args.train_N_way == self.args.test_N_way)
            num_inputs = self.emb_size + self.args.train_N_way
            self.gnn_obj = GNN_active(args, num_inputs, self.args.hidden_f, J=1)
        else:
            raise NotImplementedError

    def gnn_iclr_forward(self, z, zi_s, labels_yi):
        # Creating WW matrix
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))  #batch_size
        zero_pad = zero_pad.cuda()
        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s
        nodes = [torch.cat([zi[:,0,0,:], label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)
        W_ALL, logits = self.gnn_obj(nodes)
        logits = logits.squeeze(-1)
        outputs = F.sigmoid(logits)

        return W_ALL,outputs, logits

    def gnn_iclr_active_forward(self, z, zi_s, labels_yi, oracles_yi, hidden_layers):
        # Creating WW matrix
        zero_pad = Variable(torch.ones(labels_yi[0].size())*1.0/labels_yi[0].size(1))
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
        '''input: [batch_x, [batches_xi], [labels_yi]]'''
        [z, zi_s, labels_yi, oracles_yi] = inputs

        if 'gnn_iclr_active' in self.metric_network:
           return self.gnn_iclr_active_forward(z, zi_s, labels_yi, oracles_yi)
        elif 'gnn_iclr' in self.metric_network:
            return self.gnn_iclr_forward(z, zi_s, labels_yi)
        else:
            raise NotImplementedError


class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs)
        else:
            raise(NotImplementedError)


def load_model(model_name, args, io):
    try:
        model = torch.load('checkpoints/%s/models/%s.t7' % (args.exp_name, model_name))
        io.cprint('Loading Parameters from the last trained %s Model' % model_name)
        return model
    except:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None

class ResNet_m(nn.Module):

    def __init__(self, f):
        super(ResNet_m, self).__init__()
        self.mask = nn.Parameter(torch.ones(f),requires_grad=True)

    def forward(self, x):

        x = torch.mul(x,self.mask)
        return [self.mask,x]


def create_models(args,f):
    print (args.dataset)
    mask_data = ResNet_m(f)
    return mask_data, MetricNN(args, emb_size=f)
