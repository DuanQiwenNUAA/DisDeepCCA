from utils import load_data, svm_classify
try:
    import pickle as thepickle
except ImportError:
    import pickle as thepickle
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import numpy as np

import numpy as np
from sklearn.preprocessing import normalize

import numpy as np
from scipy.linalg import svd as scipy_svd
from collections import Counter
import scipy.io as sio
import operator
from functools import reduce

class DisCCA(): 
    def __init__(self):
        self.wy=None
        self.wy=None
        self.output_dimensions=None
        self.C_B=0
        self.C_W=0

    '''
    Preprocess X and Y to nd array and get all  tuples of same classes together
    '''
    def preprocessing(self, X, Y, target):

        target_copy=[[target[i],i] for i in range(len(target))]
        target_copy.sort()


        if(type(X).__module__!='numpy'):
            X=X.to_numpy()
        if(type(Y).__module__!='numpy'):
            Y=Y.to_numpy()

        
        X_copy=np.array([])
        for i in range(len(target_copy)):
            new_row=X[target_copy[i][1]]
            if(len(X_copy)==0):
                X_copy=[new_row]
            else:
                X_copy = np.vstack([X_copy, new_row])

        Y_copy=np.array([])
        for i in range(len(target_copy)):
            new_row=Y[target_copy[i][1]]
            if(len(Y_copy)==0):
                Y_copy=[new_row]
            else:
                Y_copy = np.vstack([Y_copy, new_row])        
    
        X=X_copy.T
        Y=Y_copy.T
        return X,Y

    '''
    Function to fit data to model
    '''

    def fit(self, X, Y, target, output_dimensions):

        self.output_dimensions=output_dimensions

        X,Y=self.preprocessing(X, Y, target)# 对原数据按类别排序
        X_shape=X.shape
        Y_shape=Y.shape
    
        #Zero mean X and Y
        X_hat=X-X.mean(axis=1, keepdims=True)
        Y_hat=Y-Y.mean(axis=1, keepdims=True)

        class_freq=dict(Counter(target))  
        N=len(target)

        '''
        Creating block diagonal matrix A
        A=[[1](n1*n1)
                    [1](n2*n2)
                            ...
                                ...
                                    ...

                                        [1](nc*nc) ]
        '''
        i=0
        A=np.array([])
        cumulative_co=0
        for c in class_freq:
            for j in range(class_freq[c]):
                new_row=np.concatenate((np.zeros(cumulative_co), np.ones(class_freq[c]), np.zeros(N-cumulative_co-class_freq[c])),axis=0)
                if(len(A)==0):
                    A=new_row
                else:
                    A = np.vstack([A, new_row])
            cumulative_co+=class_freq[c]
            i+=1
        
        self.C_W=np.matmul(np.matmul(X_hat,A),Y_hat.transpose()) #Within class similarity matrix
        self.C_B=-(self.C_W) #Between class similarity matrix

        Sigma_xy=(1.0 / (N - 1)) * self.C_W
        Sigma_yx=(1.0 / (N - 1)) * np.matmul(np.matmul(Y_hat,A),X_hat.T)


        '''
        regularizing Sigma_xx and Sigma_yy
        '''
        rx = 1e-3 #regulazisation coefficient for x 
        ry = 1e-3 #regulazisation coefficient for y
        Sigma_xx= (1.0 / (N - 1)) * np.matmul(X_hat,X_hat.T) + rx * np.identity(X_shape[0])
        Sigma_yy= (1.0 / (N - 1)) * np.matmul(Y_hat,Y_hat.T) + ry * np.identity(Y_shape[0])

        '''
        Finding inverse square root of  Sigma_xx and Sigma_yy
        using A^(-1/2)= PΛ^(-1/2)P'
        where
        P is matrix containing Eigen vectors of A in row form
        Λ is diagonal matrix containing eigen values in diagonal
        '''
        [eigen_values_xx, eigen_vectors_matrix_xx] = np.linalg.eigh(Sigma_xx)
        [eigen_values_yy, eigen_vectors_matrix_yy]= np.linalg.eigh(Sigma_yy)
        Sigma_xx_root_inverse = np.dot(np.dot(eigen_vectors_matrix_xx, np.diag(eigen_values_xx ** -0.5)), eigen_vectors_matrix_xx.T)
        Sigma_yy_root_inverse = np.dot(np.dot(eigen_vectors_matrix_yy, np.diag(eigen_values_yy ** -0.5)), eigen_vectors_matrix_yy.T)

        T=np.matmul(np.matmul(Sigma_xx_root_inverse,Sigma_xy),Sigma_yy_root_inverse)

        U, S, V = np.linalg.svd(T)
        V = V.T

        self.wx= np.dot(Sigma_xx_root_inverse, U[:, 0:self.output_dimensions])
        self.wy= np.dot(Sigma_yy_root_inverse, V[:, 0:self.output_dimensions])       
        
        return None

    '''
    transform data to new view
    '''
    def transform(self, X, Y):

        if(type(X).__module__!='numpy'):
            X=X.to_numpy()
        if(type(Y).__module__!='numpy'):
            Y=Y.to_numpy()
        
        X_transformed=np.matmul(X,self.wx)
        Y_transformed=np.matmul(Y,self.wy)
        
        return X_transformed, Y_transformed

    def get_within_class_similarity(self):
        return self.C_W

    def get_between_class_similarity(self):
        return self.C_B
    
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

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 30

    datatype = 4
        # 读取.mat文件
    if datatype == 1:
        # 六个视图，分别为：（2000，216）、（2000，76）、（2000，64）、（2000，6）、（2000，240）、（2000，47）
        mat_data = sio.loadmat(r"D:\本科毕业设计\Python_Projects\DataSets\数据集\Mfeat.mat")
        data1 = mat_data['fea'][0][1]
        data2 = mat_data['fea'][0][4]

        target = mat_data['gt']
    elif datatype == 2:
        # 四个视图，分别为：（400，512）、（400，59）、（400，864）、（400，254）
        mat_data = sio.loadmat(r"D:\本科毕业设计\Python_Projects\DataSets\数据集\ORL.mat")
        data1 = mat_data['fea'][0][0]
        data2 = mat_data['fea'][0][1]

        target = mat_data['gt']
    elif datatype == 3:
        # 两个视图，分别为：（2000，784）、（2000，256）
        mat_data = sio.loadmat(r"D:\本科毕业设计\Python_Projects\DataSets\数据集\HW22.mat")
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

    target = reduce(operator.concat, target.tolist())
    
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
        train_rate = 0.8
        val_rate = 0.1
        test_rate = 0.1
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
        setnum = 195 # 每一类的样本数
        train_rate = 0.6
        val_rate = 0.2
        test_rate = 0.2

    train1, val1, test1 = getsets(data1, train_rate, val_rate, test_rate, allnum, setnum)
    train2, val2, test2 = getsets(data2, train_rate, val_rate, test_rate, allnum, setnum)
    target_train, target_v, target_t = get_targetsets(target, train_rate, val_rate, test_rate, allnum, setnum)
    target = target_train + target_v + target_t

    #使用CCA对两视图进行降维
    v_acccca=[]
    t_acccca=[]
    
    l_cca=linear_cca()
    l_cca.fit(train1, train2, outdim_size)
    
    result_train0, result_train1=l_cca.test(train1, train2)
    result_v0, result_v1=l_cca.test(val1, val2)
    result_t0, result_t1 = l_cca.test(test1, test2)

    outputscca0 = np.concatenate((result_train0, result_v0, result_t0))
    outputscca1 = np.concatenate((result_train1, result_v1, result_t1))

    #使用DisCCA对两视图进行降维
    v_acc=[]
    t_acc=[]

    D_cca=DisCCA()
    D_cca.fit(train1, train2, target_train, outdim_size)

    result_train=D_cca.transform(train1, train2)
    result_v=D_cca.transform(val1, val2)
    result_t=D_cca.transform(test1, test2)

    outputsdcca0 = np.concatenate((result_train[0], result_v[0], result_t[0]))
    outputsdcca1 = np.concatenate((result_train[1], result_v[1], result_t[1]))


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

    vacccca1 = []
    tacccca1 = []

    vacccca2 = []
    tacccca2 = []

    print('training SVM...')
    for batch_idx in batch_idxs:
        batch_x1 = outputsdcca0[batch_idx, :]
        batch_x2 = outputsdcca1[batch_idx, :]
        batch_target = [target[i] for i in batch_idx]
        [test_acc2, valid_acc2] = svm_classify(batch_x2, batch_x1, batch_target, C=0.01)
        [test_acc1, valid_acc1] = svm_classify(batch_x1, batch_x2, batch_target, C=0.01)


        vacc1.append(valid_acc1)
        tacc1.append(test_acc1)
        vacc2.append(valid_acc2)
        tacc2.append(test_acc2)

        batch_x1 = outputscca0[batch_idx, :]
        batch_x2 = outputscca1[batch_idx, :]
        batch_target = [target[i] for i in batch_idx]
        [test_acccca1, valid_acccca1] = svm_classify(batch_x1, batch_x2, batch_target, C=0.01)
        [test_acccca2, valid_acccca2] = svm_classify(batch_x2, batch_x1, batch_target, C=0.01)

        vacccca1.append(valid_acccca1)
        tacccca1.append(test_acccca1)
        vacccca2.append(valid_acccca2)
        tacccca2.append(test_acccca2)
        

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

    vacccca = []
    tacccca = []

    c=np.concatenate((outputsdcca0 , outputsdcca1))
    ccca=np.concatenate((outputscca0 , outputscca1))
    c_target = target + target

    for batch_idx in batch_idxs:
        c_batch = c[batch_idx, :]
        c_batch_target = [c_target[i] for i in batch_idx]
        [test_acc, valid_acc] = svm_classify(c_batch, batch_x1, c_batch_target, C=0.01)

        vacc.append(valid_acc)
        tacc.append(test_acc)

        c_batch = ccca[batch_idx, :]
        c_batch_target = [c_target[i] for i in batch_idx]
        [test_acccca, valid_acccca] = svm_classify(c_batch, batch_x1, c_batch_target, C=0.01)

        vacccca.append(valid_acccca)
        tacccca.append(test_acccca)

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

    vacccca_mean = np.mean(vacccca)
    vacccca_var = np.var(vacccca)
    tacccca_mean = np.mean(tacccca)
    tacccca_var = np.var(tacccca)
    vacccca1_mean = np.mean(vacccca1)
    vacccca1_var = np.var(vacccca1)
    tacccca1_mean = np.mean(tacccca1)
    tacccca1_var = np.var(tacccca1)
    vacccca2_mean = np.mean(vacccca2)
    vacccca2_var = np.var(vacccca2)
    tacccca2_mean = np.mean(tacccca2)
    tacccca2_var = np.var(tacccca2)

    #print("Accuracy on view 1 (validation data) is:", vacc1_mean * 100.0, "+-", vacc1_var * 100.0)
    print("CCA Accuracy on view 1 (test data) is:", tacccca1_mean*100.0, "+-", tacccca1_var * 100.0)
    #print("Accuracy on view 2 (validation data) is:", vacc2_mean * 100.0, "+-", vacc2_var * 100.0)
    print("CCA Accuracy on view 2 (test data) is:", tacccca2_mean*100.0, "+-", tacccca2_var * 100.0)
    #print("Accuracy (validation data) is:", vacc_mean * 100.0, "+-", vacc_var * 100.0)
    print("CCA Accuracy (test data) is:", tacccca_mean*100.0, "+-", tacccca_var * 100.0)

        #print("Accuracy on view 1 (validation data) is:", vacc1_mean * 100.0, "+-", vacc1_var * 100.0)
    print("DisCCA Accuracy on view 1 (test data) is:", tacc1_mean*100.0, "+-", tacc1_var * 100.0)
    #print("Accuracy on view 2 (validation data) is:", vacc2_mean * 100.0, "+-", vacc2_var * 100.0)
    print("DisCCA Accuracy on view 2 (test data) is:", tacc2_mean*100.0, "+-", tacc2_var * 100.0)
    #print("Accuracy (validation data) is:", vacc_mean * 100.0, "+-", vacc_var * 100.0)
    print("DisCCA Accuracy (test data) is:", tacc_mean*100.0, "+-", tacc_var * 100.0)