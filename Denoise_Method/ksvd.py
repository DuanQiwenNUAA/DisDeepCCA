#coding=utf-8
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import normalize
import scipy.misc
from matplotlib import pyplot as plt
import random
import cv2
 
 
class KSVD(object):
    def __init__(self, k, max_iter=30, tol=1e-6,
                 n_nonzero_coefs=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecodeX = None
        self.max_iter = max_iter
        self.sigma = tol
        self.k_components =k
        self.n_nonzero_coefs = n_nonzero_coefs
 
    def _initialize(self, y):
 
        # u, s, v = np.linalg.svd(y)
        # self.dictionary = u[:, :self.k_components]
        """
        随机选取样本集y中n_components个样本,并做L2归一化
        #  """
        ids=np.arange(y.shape[1])                                     #获得列索引数组
        select_ids=random.sample(list(ids), self.k_components ) #随机选取k_components个样本的id,k-svd之K
        mid_dic=y[:,np.array(select_ids)]                           #数组切片提取出k个样本
        self.dictionary=normalize(mid_dic, axis=0, norm='l2')  #每一列做L2归一化
 
        print('字典规模：',self.dictionary.shape)
 
    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.k_components):
            index = np.nonzero(x[i, :])[0]  #非零项索引数组
            if len(index) == 0:
                continue
 
            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]                      #获取非零项对用id的列
            u, s, v = np.linalg.svd(r, full_matrices=False)  #SVD分解
            d[:, i] = u[:, 0]
            x[i, index] = s[0] * v[0, :]
        return d, x
 
    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            print('训练次数：',i,'训练误差：', e)
            if e < self.sigma:
                break
            self._update_dict(y, self.dictionary, x)
 
 
        self.sparsecodeX = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecodeX
 
 
def gaussian_noise(image):
    h, w= image.shape
    mean = 0
    sigma = 20  # 标准差
    noise = np.random.normal(mean, sigma, (h, w)) #根据均值和标准差生成符合高斯分布的噪声
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


if __name__ == '__main__':
 
    #im_ascent = scipy.misc.ascent().astype(float)
    #读入训练样本
    im_ascent=cv2.imread(r"C:\Users\Wisdom\Pictures\lena.png", 0).astype(float)
    print('训练样本规模：',im_ascent.shape)
    #进行ksvd迭代，学习字典
    ksvd = KSVD(200)
    #ksvd = KSVD(100,100,0.00005,5)
    dictionary, sparsecode = ksvd.fit(im_ascent)
    #对原图施加噪声
    im_ascent_noise=gaussian_noise(im_ascent)
    #使用字典得到噪声图的稀疏向量x
    sparsecode=linear_model.orthogonal_mp(dictionary,im_ascent_noise)
    #打印原图、噪声图、去噪图
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im_ascent)
    plt.subplot(1, 3, 2)
    plt.imshow(im_ascent_noise)
    plt.subplot(1, 3, 3)
    plt.imshow(dictionary.dot(sparsecode))
    plt.show()
    
    #保存噪声图、去噪图
    cv2.imwrite(r"D:\Python_Projects\Denoise_Method\Denoise_Method\noise_lena.png", im_ascent_noise.astype(np.uint8))

    train_restruct=dictionary.dot(sparsecode)
    train_restruct = np.clip(train_restruct, 0, 255)
    cv2.imwrite(r"D:\Python_Projects\Denoise_Method\Denoise_Method\denoise_lena.png", train_restruct.astype(np.uint8))

