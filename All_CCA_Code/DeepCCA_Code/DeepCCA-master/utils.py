import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch


def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set)
    valid_set_x, valid_set_y = make_tensor(valid_set)
    test_set_x, test_set_y = make_tensor(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def svm_classify(x1, x2, target, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """

    train_rate = 0.6
    val_rate = 0.2
    test_rate = 0.2
    allnum = x1.shape[0]
    
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(x1[0:int(allnum * train_rate), : ], target[0:int(allnum * train_rate)])

    p = clf.predict(x1[int(allnum * train_rate):int(allnum * train_rate + allnum * val_rate)])
    valid_acc = accuracy_score(target[int(allnum * train_rate):int(allnum * train_rate + allnum * val_rate)], p)
    p = clf.predict(x1[int(allnum * train_rate + allnum * val_rate):allnum, :])
    test_acc = accuracy_score(target[int(allnum * train_rate + allnum * val_rate):allnum], p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import pickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret
