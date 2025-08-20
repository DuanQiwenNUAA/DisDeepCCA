import gzip
from utils import load_data, svm_classify
import logging
try:
    import pickle as thepickle
except ImportError:
    import pickle as thepickle

data1 = load_data(r"D:\Python_Projects\DeepCCA_Code\DeepCCA_Code\DeepCCA-master\noisymnist_view1.gz")
data2 = load_data(r"D:\Python_Projects\DeepCCA_Code\DeepCCA_Code\DeepCCA-master\noisymnist_view2.gz")
print(data1[2][0][1].size())
print(data2[0][1].size)