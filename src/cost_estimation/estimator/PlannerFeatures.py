import numpy as np
from scipy.sparse import hstack
#features
#read/bandwidth f1
#write/bandwidth f2
#complexity_ratio f3
#percentage complexity scalar f4
#percentage complexity LMM f5
#percentage complexity RMM f6
#scalar flops/peak f7
#LMM flops/peak f8
#RMM flops/peak f9
#col_op/total_op f10
#tuple ratio f11
#feature ratio f12
class PlannerFeatures:
    """
    Estimates the cost of different operations and ML models for an IlargiMatrix object.
    """

    def __new__(self, AM, T):
        self.T = T
        self.AM = AM

        self.r_S = [self.AM.S[k].shape[0] for k in range(self.AM.K)]
        self.c_S = [self.AM.S[k].shape[1] for k in range(self.AM.K)]

        self.TR = self.r_S[0] / sum(self.r_S[1:])
        self.FR = sum(self.c_S[1:]) / self.c_S[0]

        return self

    def scalar_op(self):
        """
        Cost of a scalar operation on a matrix.
        """
        standard = self.T.nnz
        factorized = sum(self.AM.S[k].nnz for k in range(self.AM.K))
        return np.array([standard, factorized, \
        self.T.nnz*8, self.T.nnz*8, \
        sum(self.AM.S[k].nnz*8 for k in range(self.AM.K)),\
        sum(self.AM.S[k].nnz*8 for k in range(self.AM.K))])
    
    def LMM(self, X_shape):
        """
        Cost of a matrix multiplication between a matrix and a vector.
        """
        # TODO: flexible way
        X_nnz = X_shape[0] * X_shape[1]
        standard = X_shape[1] * self.T.nnz + self.T.shape[0] * X_nnz
        factorized = sum([X_shape[1] * self.AM.S[k].nnz + self.AM.S[k].shape[0] * X_nnz for k in range(self.AM.K)])
        # read_standard = X_nnz * 8 + self.T.nnz * 8
        read_standard=np.sqrt(X_nnz)*self.T.nnz*8
        write_standard=self.T.shape[0] * X_shape[1] * 8
        read_factorized = sum([np.sqrt(X_nnz) * self.AM.S[k].nnz * 8 for k in range(self.AM.K)])
        write_factorized = sum([self.AM.S[k].shape[0] * X_shape[1] * 8 for k in range(self.AM.K)])
        return np.array([standard, factorized, \
        read_standard, write_standard, \
        read_factorized, write_factorized])

    def plain_MM(self, A_shape,B_shape, A_nnz, B_nnz):
        standard=B_shape[1]*A_nnz+A_shape[0]*B_nnz
        standard=standard
        write_standard=A_shape[0]*B_shape[1]*8
        read_standard=(A_shape[0]*A_shape[1]*B_shape[1])*8
        return np.array([standard, standard, \
        read_standard, write_standard, \
        read_standard, write_standard])


    def LMM_T(self, X_shape):
        """
        Cost of a matrix multiplication between a vector and a matrix.
        """
        X_nnz = X_shape[0] * X_shape[1]
        standard = self.T.shape[1] * X_nnz + X_shape[1] * self.T.nnz
        factorized = sum([self.AM.S[k].shape[1] * X_nnz + X_shape[1] * self.AM.S[k].nnz for k in range(self.AM.K)])
        read_standard =np.sqrt(X_nnz) * self.T.nnz * 8
        write_standard = self.T.shape[1]*X_shape[1]*8
        read_factorized = sum([np.sqrt(X_nnz) * self.AM.S[k].nnz * 8 for k in range(self.AM.K)])
        write_factorized=sum([self.AM.S[k].shape[1] * X_shape[1] * 8 for k in range(self.AM.K)])
        return np.array([standard, factorized, \
        read_standard, write_standard, \
        read_factorized, write_factorized])

    def RMM(self, X_shape):
        """
        Cost of a matrix multiplication between a matrix and a vector.
        """
        X_nnz = X_shape[0] * X_shape[1]
        standard = self.T.shape[1] * X_nnz + X_shape[0] * self.T.nnz
        factorized = sum([self.AM.S[k].shape[1] * X_nnz + X_shape[0] * self.AM.S[k].nnz for k in range(self.AM.K)])
        read_standard = X_nnz * np.sqrt(self.T.nnz) * 8
        write_standard = X_shape[0]*self.T.shape[1]*8
        read_factorized= sum([X_nnz * np.sqrt(self.AM.S[k].nnz) * 8 for k in range(self.AM.K)])
        write_factorized=sum([self.AM.S[k].shape[1] * X_shape[0] * 8 for k in range(self.AM.K)])
        return np.array([standard, factorized, \
        read_standard, write_standard, \
        read_factorized, write_factorized])
    def RMM_T(self, X_shape):
        """
        Cost of a matrix multiplication between a vector and a matrix.
        """
        X_nnz = X_shape[0] * X_shape[1]
        standard = X_shape[0] * self.T.nnz + self.T.shape[0] * X_nnz
        factorized = sum([X_shape[0] * self.AM.S[k].nnz + self.AM.S[k].shape[0] * X_nnz for k in range(self.AM.K)])
        read_standard = X_nnz * np.sqrt(self.T.nnz) * 8
        write_standard = X_shape[0]*self.T.shape[0]*8
        read_factorized= sum([X_nnz * np.sqrt(self.AM.S[k].nnz) * 8 for k in range(self.AM.K)])
        write_factorized=sum([self.AM.S[k].shape[0] * X_shape[0] * 8 for k in range(self.AM.K)])
        return np.array([standard, factorized, \
        read_standard, write_standard, \
        read_factorized, write_factorized])
    
    def element_wise(self, shape):

        return np.array([shape[0]*shape[1] , shape[0]*shape[1],\
        shape[0]*shape[1]*8 , shape[0]*shape[1]*8,\
        shape[0]*shape[1]*8 , shape[0]*shape[1]*8])
    
    
    def MorpheusEst(self):
        """
        Cost estimated by Morpheus.
        """
        tau = 5
        rho = 1
        if self.TR < tau or self.FR < rho:
            # No factorization
            return False
        else:
            # Factorization
            return True
        
    def extract_features(self, model,features_list):
        total_ma_complexity=np.add.reduce([x[0] for x in features_list])
        total_fa_complexity=np.add.reduce([x[1] for x in features_list])
        f=np.zeros(28)
        
        if model=='LinR':
            f[0]=np.add.reduce([x[2] for x in features_list[2:]]) #ma read
            f[1]=np.add.reduce([x[3] for x in features_list[2:]]) #ma write
            f[2]=np.add.reduce([x[4] for x in features_list[2:]]) #fa read
            f[3]=np.add.reduce([x[5] for x in features_list[2:]]) #fa write
            f[4]=0#features_list[0][0]+features_list[1][0] #scalar ma
            f[5]=np.add.reduce([x[0] for x in features_list[2:]]) #lmm ma
            f[6]=0 #rmm ma
            f[7]=0 #scalar fa
            f[8]=np.add.reduce([x[1] for x in features_list[2:]]) #lmm fa
            f[9]=0 #rmm fa
            f[10]=features_list[3][0] #col ma
            f[11]=features_list[3][1] #col fa
            f[12]=features_list[0][0]+features_list[1][0] #dense scalar complexity
            f[13]=0 #dense MM complexity
            f[14]=features_list[0][2]+features_list[1][2] #dense scalar read
            f[15]=features_list[0][3]+features_list[1][3] #dense scalar write
            f[16]=0 #dense MM read
            f[17]=0 #dense MM write
            f[18]=0 #rowsum complexity
            f[19]=0 #rowsum read
            f[20]=0 #rowsum write
            f[21]=0 #colsum complexity
            f[22]=0 #colsum read
            f[23]=0 #colsum write

        elif model=='LogR':
            f[0]=np.add.reduce([x[2] for x in features_list[2:]]) #ma read
            f[1]=np.add.reduce([x[3] for x in features_list[2:]]) #ma write
            f[2]=np.add.reduce([x[4] for x in features_list[2:]]) #fa read
            f[3]=np.add.reduce([x[5] for x in features_list[2:]]) #fa write
            f[4]=0#features_list[0][0]+features_list[1][0] #scalar ma
            f[5]=np.add.reduce([x[0] for x in features_list[2:]]) #lmm ma
            f[6]=0 #rmm ma
            f[7]=0 #scalar fa
            f[8]=np.add.reduce([x[1] for x in features_list[2:]]) #lmm fa
            f[9]=0 #rmm fa
            f[10]=features_list[3][0] #col ma
            f[11]=features_list[3][1] #col fa
            f[12]=features_list[0][0]+features_list[1][0] #dense scalar complexity
            f[13]=0 #dense MM complexity
            f[14]=features_list[0][2]+features_list[1][2] #dense scalar read
            f[15]=features_list[0][3]+features_list[1][3] #dense scalar write
            f[16]=0 #dense MM read
            f[17]=np.add.reduce([features_list[x][3] for x in []]) #dense MM write

        elif model=='KMeans':
            f[0]=np.add.reduce([features_list[x][2] for x in [0,2,10]]) #ma read
            f[1]=np.add.reduce([features_list[x][3] for x in [0,2,10]]) #ma write
            f[2]=np.add.reduce([features_list[x][4] for x in [0,2,10]]) #fa read
            f[3]=np.add.reduce([features_list[x][5] for x in [0,2,10]]) #fa write
            f[4]=np.add.reduce([features_list[x][0] for x in [0]])#scalar ma
            f[5]=np.add.reduce([features_list[x][0] for x in [2,10]]) #lmm ma
            # f[6]=np.add.reduce([features_list[x][0] for x in []]) #rmm ma
            f[7]=np.add.reduce([features_list[x][1] for x in [0]]) #scalar fa
            f[8]=np.add.reduce([features_list[x][1] for x in [2,10]]) #lmm fa
            # f[9]=np.add.reduce([features_list[x][1] for x in []]) #rmm fa
            f[10]=np.add.reduce([features_list[x][0] for x in [10]]) #col ma
            f[11]=np.add.reduce([features_list[x][1] for x in [10]]) #col fa
            f[12]=np.add.reduce([features_list[x][0] for x in [3,6,13]]) #dense scalar complexity
            f[13]=np.add.reduce([features_list[x][0] for x in [1,5,9,12]]) #dense MM complexity
            f[14]=np.add.reduce([features_list[x][2] for x in [3,6,13]]) #dense scalar read
            f[15]=np.add.reduce([features_list[x][3] for x in [3,6,13]]) #dense scalar write
            f[16]=np.add.reduce([features_list[x][2] for x in [1,5,9,12]]) #dense MM read
            f[17]=np.add.reduce([features_list[x][3] for x in [1,5,9,12]]) #dense MM write
            f[18]=np.add.reduce([features_list[x][0] for x in [7,8]]) #rowsum complexity
            f[19]=np.add.reduce([features_list[x][2] for x in [7,8]]) #rowsum read
            f[20]=np.add.reduce([features_list[x][3] for x in [7,8]]) #rowsum write
            f[21]=np.add.reduce([features_list[x][0] for x in [4,11]]) #colsum complexity
            f[22]=np.add.reduce([features_list[x][2] for x in [4,11]]) #colsum read
            f[23]=np.add.reduce([features_list[x][3] for x in [4,11]]) #colsum write


        elif model=='GaussianNMF':
            f[0]=np.add.reduce([features_list[x][2] for x in [0,4]]) #ma read
            f[1]=np.add.reduce([features_list[x][3] for x in [0,4]]) #ma write
            f[2]=np.add.reduce([features_list[x][4] for x in [0,4]]) #fa read
            f[3]=np.add.reduce([features_list[x][5] for x in [0,4]]) #fa write
            f[12]=np.add.reduce([features_list[x][0] for x in [3,7]]) #dense scalar complexity
            f[13]=np.add.reduce([features_list[x][0] for x in [1,2,5,6]]) #dense MM complexity
            f[14]=np.add.reduce([features_list[x][2] for x in [3,7]]) #dense scalar read
            f[15]=np.add.reduce([features_list[x][3] for x in [3,7]]) #dense scalar write
            f[16]=np.add.reduce([features_list[x][2] for x in [1,2,5,6]]) #dense MM read
            f[17]=np.add.reduce([features_list[x][3] for x in [1,2,5,6]]) #dense MM write

        f[24]=total_ma_complexity
        f[25]=total_fa_complexity
        f[26]=self.TR
        f[27]=self.FR
        return f
            


    def LinR(self):
        """
        Cost of a linear regression model.
        """
        X_shape = [self.T.shape[1], 1]
        s1 = self.element_wise(self, (self.T.shape[1], 1))
        s1 = s1*2
        s2=self.element_wise(self, (self.T.shape[0], 1))
        s3 = self.LMM_T(self, (self.T.shape[0], 1))
        s4 = self.LMM(self, X_shape)
        return self.extract_features(self,'LinR',[s1,s2,s3,s4])
        

    def LogR(self):
        """
        Cost of a logistic regression model.
        """
        # TODO: correct?
        X_shape = [self.T.shape[1], 1]
        s1 = self.element_wise(self, (self.T.shape[0], 1))
        s2 = self.element_wise(self, (self.T.shape[0], 1))
        s1=s1*3
        s2=s2*2
        s3 = self.LMM(self, X_shape)
        s4 = self.LMM_T(self, (self.T.shape[0], 1))
        return self.extract_features(self,'LogR',[s1,s2,s3,s4])

    def KMeans(self, k, iterations):
        """
        Cost of a K-means model.
        """
        C_shape = [self.T.shape[1], k]
        A_shape = [self.T.shape[0], k]
        s1=self.scalar_op(self)
        s1=s1*2/iterations
        #miss rowsum(T^2)
        s2=self.plain_MM(self,(self.T.shape[0],1),(1,k),self.T.nnz,k)#not clear of vector multiplication with 1xk
        s2=s2/iterations
        s3=self.LMM(self,C_shape) #T2C
        s4=self.element_wise(self,C_shape) #C^2\
        s5=(C_shape[0]*C_shape[1],C_shape[0]*C_shape[1],C_shape[0]*C_shape[1]*8,C_shape[0]*C_shape[1]*8,C_shape[1]*8,C_shape[1]*8) #colsum(c^2)
        s6=self.plain_MM(self,(self.T.shape[0],1),(1,k),self.T.shape[0],k) #1Xcolsum(c^2)
        s7=self.element_wise(self,(self.T.shape[0],k)) #dt-t2c+1Xcolsum(c^2)
        s7=s7*2
        s8=(self.T.shape[0]*k,self.T.shape[0]*k,self.T.shape[0]*k*8,self.T.shape[0]*k*8,k*8,k*8) #rowmin
        s9=(self.T.shape[0]*k,self.T.shape[0]*k,self.T.shape[0]*k*8,self.T.shape[0]*k*8,self.T.shape[0]*8,self.T.shape[0]*8) #D==rowmin
        s10=self.plain_MM(self,(self.T.shape[0],1),(1,k),self.T.shape[0],k) #A
        s11 = self.LMM_T(self, A_shape) #TTA
        s12=(self.T.shape[0]*k,self.T.shape[0]*k,self.T.shape[0]*k*8,self.T.shape[0]*k*8,k*8,k*8) #colsum A
        s13=self.plain_MM(self,(self.T.shape[1],1),(1,k),self.T.shape[1],k) #1Xcolsum(A)
        s14=self.element_wise(self,(self.T.shape[1],k)) #update C
        return self.extract_features(self,'KMeans',[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14])

    def GaussianNMF(self, r):
        """
        Cost of a Gaussian NMF model.
        """
        W_shape = [self.AM.shape[0], r]
        H_shape = [r, self.AM.shape[1]]
        s1=self.RMM(self, (W_shape[1], W_shape[0]))
        s2=self.plain_MM(self, (W_shape[1], W_shape[0]), (W_shape[0], W_shape[1]), W_shape[0]*W_shape[1], W_shape[1]*W_shape[0])
        s3=self.plain_MM(self,(r,r),(r,H_shape[1]),r*r,r*H_shape[1])
        s4=self.element_wise(self,(r,H_shape[1]))
        
        s5=self.LMM(self, (H_shape[1], H_shape[0]))
        s6=self.plain_MM(self, (H_shape[0], H_shape[1]), (H_shape[1], H_shape[0]), H_shape[0]*H_shape[1], H_shape[1]*H_shape[0])
        s7=self.plain_MM(self,(W_shape[0],r),(r,r),W_shape[0]*r,r*r)
        s8=self.element_wise(self,(W_shape[0],r))
        return self.extract_features(self,'GaussianNMF',[s1,s2,s3,s4,s5,s6,s7,s8])
