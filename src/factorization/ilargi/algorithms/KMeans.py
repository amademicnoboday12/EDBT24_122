from factorization.ilargi.IlargiMatrix import csp, xp
import scipy.sparse as sp


class KMeans:
    def __new__(self, K, X, iterations):
        self.K = K
        self.iterations = iterations
        self.r_X, self.c_X = X.shape[0], X.shape[1]
        # Initalize centroids matrix C
        # TODO: smart way of initializing centroids (without use of T)
        self.C = sp.dok_matrix((self.c_X, self.K))
        self.C[:] = 1
        self.C = csp.csr_matrix(self.C)

        return self

    def fit(self, X):
        # Pre-compute l^2-norm of points for distance
        self.D_T = (X.power(2)).sum(axis=1).reshape(-1, 1) * xp.ones((1, self.K))
        self.T_2 = X * 2

        for _ in range(self.iterations):
            # C_start = C
            # Compute pairwise squared distances; D has points on the rows and centroids / clusters on the columns
            D = (
                    self.D_T
                    - self.T_2.dot(self.C)
                    + xp.ones((self.r_X, 1)) * (self.C.power(2)).sum(axis=0)
            )
            # Assign each point to nearest centroid
            A = csp.csr_matrix(xp.multiply((D == D.min(axis=1).reshape(-1, 1)), xp.ones((self.r_X, self.K))))
            # Compute new centroids; the denominator counts the number of points in the new clusters, while the numerator adds up assigned points per cluster
            self.C = csp.csr_matrix((X.T.dot(A)) / (xp.ones((self.c_X, 1)) * A.sum(axis=0)))
            # Stop iterating in case clusters have converged
            # if (C == C_start).all():
            #     break
        self.D = D

    def predict(self):
        # Assign each point to nearest centroid
        A = self.D == self.D.min(axis=1).reshape(-1, 1) * xp.ones((self.r_X, self.K))
        labels = xp.where(A)[1]
        return labels
