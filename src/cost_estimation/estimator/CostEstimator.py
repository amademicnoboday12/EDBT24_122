import numpy as np
from scipy.sparse import hstack

class CostEstimator:
    """
    Estimates the cost of different operations and ML models for an IlargiMatrix object.
    """

    def __new__(self, AM):
        self.T = AM.materialize()
        self.AM = AM

        self.r_S = [self.AM.S[k].shape[0] for k in range(self.AM.K)]
        self.c_S = [self.AM.S[k].shape[1] for k in range(self.AM.K)]

        self.TR = self.r_S[0] / sum(self.r_S[1:])
        self.FR = sum(self.c_S[1:]) / self.c_S[0]
        self.cardinality_T = self.T.shape[0]
        self.cardinality_S = sum(self.r_S)
        self.selectivity = self.cardinality_S / self.cardinality_T

        return self

    def scalar_op(self):
        """
        Cost of a scalar operation on a matrix.
        """
        standard = self.T.nnz
        factorized = sum(self.AM.S[k].nnz for k in range(self.AM.K))
        return standard, factorized
    
    def LMM(self, X_shape):
        """
        Cost of a matrix multiplication between a matrix and a vector.
        """
        # TODO: flexible way
        X_nnz = X_shape[0] * X_shape[1]
        standard = X_shape[1] * self.T.nnz + self.T.shape[0] * X_nnz
        factorized = sum([X_shape[1] * self.AM.S[k].nnz + self.AM.S[k].shape[0] * X_nnz for k in range(self.AM.K)])
        return standard, factorized

    def LMM_T(self, X_shape):
        """
        Cost of a matrix multiplication between a vector and a matrix.
        """
        X_nnz = X_shape[0] * X_shape[1]
        standard = self.T.shape[1] * X_nnz + X_shape[1] * self.T.nnz
        factorized = sum([self.AM.S[k].shape[1] * X_nnz + X_shape[1] * self.AM.S[k].nnz for k in range(self.AM.K)])
        return standard, factorized

    def RMM(self, X_shape):
        """
        Cost of a matrix multiplication between a matrix and a vector.
        """
        X_nnz = X_shape[0] * X_shape[1]
        standard = self.T.shape[1] * X_nnz + X_shape[0] * self.T.nnz
        factorized = sum([self.AM.S[k].shape[1] * X_nnz + X_shape[0] * self.AM.S[k].nnz for k in range(self.AM.K)])
        return standard, factorized

    def RMM_T(self, X_shape):
        """
        Cost of a matrix multiplication between a vector and a matrix.
        """
        X_nnz = X_shape[0] * X_shape[1]
        standard = X_shape[0] * self.T.nnz + self.T.shape[0] * X_nnz
        factorized = sum([X_shape[0] * self.AM.S[k].nnz + self.AM.S[k].shape[0] * X_nnz for k in range(self.AM.K)])
        return standard, factorized

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

    def LinR(self):
        """
        Cost of a linear regression model.
        """
        X_shape = [self.T.shape[1], 1]
        standard1, factorized1 = self.LMM(self, X_shape)
        standard2, factorized2 = self.LMM_T(self, (self.T.shape[0], 1))
        standard = standard1 + standard2
        factorized = factorized1 + factorized2
        return standard, factorized

    def LogR(self):
        """
        Cost of a logistic regression model.
        """
        # TODO: correct?
        X_shape = [self.T.shape[1], 1]
        standard1, factorized1 = self.LMM(self, X_shape)
        standard2, factorized2 = self.LMM_T(self, (self.T.shape[0], 1))
        standard = standard1 + standard2
        factorized = factorized1 + factorized2
        return standard, factorized

    def KMeans(self, k):
        """
        Cost of a K-means model.
        """
        C_shape = [self.T.shape[1], k]
        A_shape = [self.T.shape[0], k]
        standard1, factorized1 = self.scalar_op(self)
        standard2, factorized2 = self.scalar_op(self)
        standard3, factorized3 = self.LMM(self, C_shape)
        standard4, factorized4 = self.LMM_T(self, A_shape)
        standard = standard1 + standard2 + standard3 + standard4
        factorized = factorized1 + factorized2 + factorized3 + factorized4
        return standard, factorized

    def GaussianNMF(self, r):
        """
        Cost of a Gaussian NMF model.
        """
        W_shape = [self.AM.shape[0], r]
        H_shape = [r, self.AM.shape[1]]
        standard1, factorized1 = self.RMM(self, (W_shape[1], W_shape[0]))
        standard2, factorized2 = self.LMM(self, (H_shape[1], H_shape[0]))
        standard = standard1 + standard2
        factorized = factorized1 + factorized2
        return standard, factorized


        



