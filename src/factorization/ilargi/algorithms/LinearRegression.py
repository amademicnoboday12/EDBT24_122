from factorization.ilargi.IlargiMatrix import csp, xp


class LinearRegression:
    def __new__(self, gamma, iterations):
        self.gamma = gamma
        self.iterations = iterations
        return self

    def fit(self, X, Y):
        self.w = csp.csr_matrix(xp.random.randn(X.shape[1], 1))
        for _ in range(self.iterations):
            self.w -= self.gamma * (X.T * (X * self.w - Y))
        return self

    def predict(self, X):
        return X * self.w