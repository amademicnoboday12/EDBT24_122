from factorization.ilargi.IlargiMatrix import csp


class GaussianNMF:
    def __new__(self, r, iterations):
        self.r = r
        self.iterations = iterations
        return self

    def fit(self, X):
        """
        Decompose X into W*H
        """
        # Initial W and H are random [0,1]
        W = csp.random(X.shape[0], self.r, format="csr")
        H = csp.random(self.r, X.shape[1], format="csr")
        for _ in range(self.iterations):
            H.multiply(csp.csr_matrix(W.T * X) / (W.T * W * H))
            W.multiply(csp.csr_matrix(X * H.T) / (W * H * H.T))
        self.W = W
        self.H = H
        return self
