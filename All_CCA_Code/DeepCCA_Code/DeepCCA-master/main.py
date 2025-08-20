import torch
import torch.nn as nn
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import load_data, svm_classify
import time
import logging
try:
    import pickle as thepickle
except ImportError:
    import pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)

import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

import scipy.io as sio
import operator
from functools import reduce

class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss, _ = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss, _ = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss, _ = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses), outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs
    
    def predict(self, x1, x2):
        outputs1 = []
        outputs2 = []
        o1, o2 = self.model(x1, x2)
        outputs1.append(o1)
        outputs2.append(o2)
        outputs = [torch.cat(outputs1, dim=0).detach().cpu().numpy(),
                   torch.cat(outputs2, dim=0).detach().cpu().numpy()]
        return outputs
    
def getsets(data, train_rate, val_rate, test_rate, allnum, setnum):
    data = data.tolist()
    train = []
    val = []
    test = []
    for i in range(0, allnum, setnum):
        train += (data[i:int(i+setnum*train_rate)])
        val += (data[i+int(setnum*train_rate):i+int(setnum*(train_rate+val_rate))])
        test += (data[i+int(setnum*(train_rate+val_rate)):i+setnum])
    return np.array(train), np.array(val), np.array(test)

def get_targetsets(data, train_rate, val_rate, test_rate, allnum, setnum):
    train = []
    val = []
    test = []
    for i in range(0, allnum, setnum):
        train += (data[i:i+int(setnum*train_rate)])
        val += (data[i+int(setnum*train_rate):i+int(setnum*(train_rate+val_rate))])
        test += (data[i+int(setnum*(train_rate+val_rate)):i+setnum])
    return train, val, test

if __name__ == '__main__':
    ############
    # Parameters Section

    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    datatype = 1
    # 读取.mat文件
    if datatype == 1:
        # 六个视图，分别为：（2000，216）、（2000，76）、（2000，64）、（2000，6）、（2000，240）、（2000，47）
        mat_data = sio.loadmat("D:\本科毕业设计\Python_Projects\DataSets\数据集\Mfeat.mat")
        data1 = mat_data['fea'][0][0]
        data2 = mat_data['fea'][0][3]

        target = mat_data['gt']
    elif datatype == 2:
        # 四个视图，分别为：（400，512）、（400，59）、（400，864）、（400，254）
        mat_data = sio.loadmat("D:\本科毕业设计\Python_Projects\DataSets\数据集\ORL.mat")
        data1 = mat_data['fea'][0][1]
        data2 = mat_data['fea'][0][3]

        target = mat_data['gt']
    elif datatype == 3:
        # 两个视图，分别为：（2000，784）、（2000，256）
        mat_data = sio.loadmat("D:\本科毕业设计\Python_Projects\DataSets\数据集\HW22.mat")
        data1 = mat_data['fea'][0][0]
        data2 = mat_data['fea'][1][0]
        target = mat_data['gt']
    
    elif datatype == 4:
        # 三个视图，分别为：（165，4096）、（165，3304）、（165，6750）
        mat_data = sio.loadmat(r"D:\本科毕业设计\Python_Projects\DataSets\数据集\yale.mat")
        data1 = mat_data['fea'][0][1].T
        data2 = mat_data['fea'][0][2].T
        target = mat_data['gt']

    elif datatype == 5:
        # 两个视图，分别为：（195，195）、（195，1703）
        mat_data = sio.loadmat(r"D:\本科毕业设计\Python_Projects\DataSets\数据集\Cornell.mat")
        data1 = mat_data['fea'][0][0].T
        data2 = mat_data['fea'][0][1].T
        target = mat_data['gt']

    
    #将标签数据转换为列表
    target = reduce(operator.concat, target.tolist())

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 6

    # 将数据打乱
     # 按比例将data1、data2分为训练集、交叉验证集和测试集
    allnum = data1.shape[0]# 总的样本数
    if datatype == 1:
        setnum = 200 # 每一类的样本数
        train_rate = 0.6
        val_rate = 0.2
        test_rate = 0.2
    elif datatype == 2:
        setnum = 10 # 每一类的样本数
        train_rate = 0.6
        val_rate = 0.2
        test_rate = 0.2
    elif datatype == 3:
        setnum = 200 # 每一类的样本数
        train_rate = 0.6
        val_rate = 0.2
        test_rate = 0.2
    elif datatype == 4:
        setnum = 11 # 每一类的样本数
        train_rate = 0.6
        val_rate = 0.2
        test_rate = 0.2
    elif datatype == 5:
        setnum = 40 # 每一类的样本数
        train_rate = 0.6
        val_rate = 0.2
        test_rate = 0.2

    train1, val1, test1 = getsets(data1, train_rate, val_rate, test_rate, allnum, setnum)
    train2, val2, test2 = getsets(data2, train_rate, val_rate, test_rate, allnum, setnum)
    target_train, target_v, target_t = get_targetsets(target, train_rate, val_rate, test_rate, allnum, setnum)
    target = target_train + target_v + target_t

    train1 = torch.from_numpy(train1)
    train2 = torch.from_numpy(train2)
    train1 = train1.to(torch.float64)
    train2 = train2.to(torch.float64)

    val1 = torch.from_numpy(val1)
    val2 = torch.from_numpy(val2)
    val1 = val1.to(torch.float64)
    val2 = val2.to(torch.float64)

    test1 = torch.from_numpy(test1)
    test2 = torch.from_numpy(test2)
    test1 = test1.to(torch.float64)
    test2 = test2.to(torch.float64)

    # size of the input for view 1 and view 2
    input_shape1 = data1.shape[1]
    input_shape2 = data2.shape[1]

    # number of layers with nodes in each one
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]

    # the parameters for training the network
    learning_rate = 1e-4
    epoch_num = 150
    batch_size = 2000

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = False
    # end of parameters section
    ############

    # Building, training, and producing the new features by DCCA
    model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                    input_shape2, outdim_size, use_all_singular_values, device=device).double()
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)
    
    solver.fit(train1, train2, val1, val2, test1, test2)

    result_train = solver.predict(train1, train2)
    result_val = solver.predict(val1, val2)
    result_test = solver.predict(test1, test2)

    outputs0 = np.concatenate((result_train[0], result_val[0], result_test[0]))
    outputs1 = np.concatenate((result_train[1], result_val[1], result_test[1]))

    # Training and testing of SVM with linear kernel on the view 1 with new features
    # 循环求均值和方差
    if datatype == 1:
        batch_size = 200
    elif datatype == 2:
        batch_size = 200
    elif datatype == 3:
        batch_size = 200
    elif datatype == 4:
        batch_size = 100
    elif datatype == 5:
        batch_size = 50

    batch_idxs = list(BatchSampler(RandomSampler(
                range(allnum)), batch_size=batch_size, drop_last=False))
    vacc1 = []
    tacc1 = []

    vacc2 = []
    tacc2 = []

    print('training SVM...')
    for batch_idx in batch_idxs:
        batch_x1 = outputs0[batch_idx, :]
        batch_x2 = outputs1[batch_idx, :]
        batch_target = [target[i] for i in batch_idx]
        [test_acc1, valid_acc1] = svm_classify(batch_x1, batch_x2, batch_target, C=0.01)
        [test_acc2, valid_acc2] = svm_classify(batch_x2, batch_x1, batch_target, C=0.01)

        vacc1.append(valid_acc1)
        tacc1.append(test_acc1)
        vacc2.append(valid_acc2)
        tacc2.append(test_acc2)
        

    if datatype == 1:
        batch_size = 200*2
    elif datatype == 2:
        batch_size = 200*2
    elif datatype == 3:
        batch_size = 200*2
    elif datatype == 4:
        batch_size = 100*2
    elif datatype == 5:
        batch_size = 50*2

    batch_idxs = list(BatchSampler(RandomSampler(
                range(allnum*2)), batch_size=batch_size, drop_last=False))
    vacc = []
    tacc = []

    c=np.concatenate((outputs0 , outputs1))
    c_target = target + target

    for batch_idx in batch_idxs:
        c_batch = c[batch_idx, :]
        c_batch_target = [c_target[i] for i in batch_idx]
        [test_acc, valid_acc] = svm_classify(c_batch, batch_x1, c_batch_target, C=0.01)

        vacc.append(valid_acc)
        tacc.append(test_acc)

    vacc_mean = np.mean(vacc)
    vacc_var = np.var(vacc)
    tacc_mean = np.mean(tacc)
    tacc_var = np.var(tacc)
    vacc1_mean = np.mean(vacc1)
    vacc1_var = np.var(vacc1)
    tacc1_mean = np.mean(tacc1)
    tacc1_var = np.var(tacc1)
    vacc2_mean = np.mean(vacc2)
    vacc2_var = np.var(vacc2)
    tacc2_mean = np.mean(tacc2)
    tacc2_var = np.var(tacc2)

    #print("Accuracy on view 1 (validation data) is:", vacc1_mean * 100.0, "+-", vacc1_var * 100.0)
    print("Accuracy on view 1 (test data) is:", tacc1_mean*100.0, "+-", tacc1_var * 100.0)
    #print("Accuracy on view 2 (validation data) is:", vacc2_mean * 100.0, "+-", vacc2_var * 100.0)
    print("Accuracy on view 2 (test data) is:", tacc2_mean*100.0, "+-", tacc2_var * 100.0)
    print("Accuracy (test data) is:", tacc_mean*100.0, "+-", tacc_var * 100.0)
    d = torch.load('checkpoint.model')
    solver.model.load_state_dict(d)
    solver.model.parameters()


