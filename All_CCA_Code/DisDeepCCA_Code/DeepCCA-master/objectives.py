import torch
import numpy as np
from scipy.linalg import svd as scipy_svd
from collections import Counter


class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def preprocessing(self, X, Y, target):

        target_copy=[[target[i],i] for i in range(len(target))]
        target_copy.sort()

        X_copy=np.array([])
        X_copy = torch.from_numpy(X_copy)
        X_copy = X_copy.cuda(0)
        for i in range(len(target_copy)):
            indices = torch.tensor([target_copy[i][1]]).cuda(0)
            new_row = torch.index_select(X, 0, indices)
            X_copy = torch.cat((X_copy, new_row), dim = 0)

        Y_copy=np.array([])
        Y_copy = torch.from_numpy(Y_copy)
        Y_copy = Y_copy.cuda(0)
        for i in range(len(target_copy)):
            indices = torch.tensor([target_copy[i][1]]).cuda(0)
            new_row = torch.index_select(Y, 0, indices)
            Y_copy = torch.cat((Y_copy, new_row), dim = 0)      
    
        X=X_copy.t()
        Y=Y_copy.t()
        return X,Y

    def loss(self, H1, H2, target):
        """
        It is the loss function of Discriminal CCA as introduced in the original paper. There can be other formulations.
        """
        # eps = 1e-9
        # UPLO = "U"
        # '''
        # Preprocess X and Y to nd array and get all  tuples of same classes together
        # '''
        # target_copy=[[target[i],i] for i in range(len(target))]
        # target_copy.sort()

        # if(type(X).__module__!='numpy'):
        #     X=X.data.cpu().numpy()
        # if(type(Y).__module__!='numpy'):
        #     Y=Y.data.cpu().numpy()

        
        # X_copy=np.array([])
        # for i in range(len(target_copy)):
        #     new_row=X[target_copy[i][1]]
        #     if(len(X_copy)==0):
        #         X_copy=[new_row]
        #     else:
        #         X_copy = np.vstack([X_copy, new_row])

        # Y_copy=np.array([])
        # for i in range(len(target_copy)):
        #     new_row=Y[target_copy[i][1]]
        #     if(len(Y_copy)==0):
        #         Y_copy=[new_row]
        #     else:
        #         Y_copy = np.vstack([Y_copy, new_row])        
    
        # X=X_copy.T
        # Y=Y_copy.T

        # '''
        # fit data to model
        # '''

        # X_shape=X.shape
        # Y_shape=Y.shape
    
        # #Zero mean X and Y
        # X_hat=X-X.mean(axis=1, keepdims=True)
        # Y_hat=Y-Y.mean(axis=1, keepdims=True)

        # class_freq=dict(Counter(target))  
        # N=len(target)

        # '''
        # Creating block diagonal matrix A
        # A=[[1](n1*n1)
        #             [1](n2*n2)
        #                     ...
        #                         ...
        #                             ...

        #                                 [1](nc*nc) ]
        # '''
        # i=0
        # A=np.array([])
        # cumulative_co=0
        # for c in class_freq:
        #     for j in range(class_freq[c]):
        #         new_row=np.concatenate((np.zeros(cumulative_co), np.ones(class_freq[c]), np.zeros(N-cumulative_co-class_freq[c])),axis=0)
        #         if(len(A)==0):
        #             A=new_row
        #         else:
        #             A = np.vstack([A, new_row])
        #     cumulative_co+=class_freq[c]
        #     i+=1
        
        # self.C_W=np.matmul(np.matmul(X_hat,A),Y_hat.transpose()) #Within class similarity matrix
        # self.C_B=-(self.C_W) #Between class similarity matrix

        # Sigma_xy=self.C_W/(N-1)
        # Sigma_yx=np.matmul(np.matmul(Y_hat,A),X_hat.T)/(N-1)


        # '''
        # regularizing Sigma_xx and Sigma_yy
        # '''
        # rx = 1e-4 #regulazisation coefficient for x 
        # ry = 1e-4 #regulazisation coefficient for y
        # Sigma_xx=(1.0 / (N - 1)) * np.matmul(X_hat,X_hat.T) + rx * np.identity(X_shape[0])
        # Sigma_yy=(1.0 / (N - 1)) * np.matmul(Y_hat,Y_hat.T) + ry * np.identity(Y_shape[0])

        # '''
        # Finding inverse square root of  Sigma_xx and Sigma_yy
        # using A^(-1/2)= PΛ^(-1/2)P'
        # where
        # P is matrix containing Eigen vectors of A in row form
        # Λ is diagonal matrix containing eigen values in diagonal
        # '''
        # [eigen_values_xx, eigen_vectors_matrix_xx] = np.linalg.eigh(Sigma_xx)
        # [eigen_values_yy, eigen_vectors_matrix_yy]= np.linalg.eigh(Sigma_yy)
        # Sigma_xx_root_inverse = np.dot(np.dot(eigen_vectors_matrix_xx, np.diag(eigen_values_xx ** -0.5)), eigen_vectors_matrix_xx.T)
        # Sigma_yy_root_inverse = np.dot(np.dot(eigen_vectors_matrix_yy, np.diag(eigen_values_yy ** -0.5)), eigen_vectors_matrix_yy.T)

        # T=np.matmul(np.matmul(Sigma_xx_root_inverse,Sigma_xy),Sigma_yy_root_inverse)
        # Tval=torch.from_numpy(T)
        # Tval = Tval.cuda(0)

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-9

        H1, H2 = self.preprocessing(H1, H2, target)# 对原数据按类别排序

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        # 求矩阵A
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

        A = torch.from_numpy(A)
        A = A.cuda(0)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(torch.matmul(H1bar,A), H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat11 = SigmaHat11 + 1e-8 * torch.randn_like(SigmaHat11)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        SigmaHat22 = SigmaHat22 + 1e-8 * torch.randn_like(SigmaHat22)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        UPLO = "U"
        [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO=UPLO)
        [D2, V2] = torch.linalg.eigh(SigmaHat22, UPLO=UPLO)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.linalg.eigh(trace_TT, UPLO=UPLO)
            U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr
