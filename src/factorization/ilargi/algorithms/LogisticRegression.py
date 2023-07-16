from factorization.ilargi.IlargiMatrix import csp, xp


class LogisticRegression:
    def __new__(self, gamma, iterations):
        self.gamma = gamma
        self.iterations = iterations
        return self

    def fit(self, X, Y):
        self.w = csp.csr_matrix(xp.random.randn(X.shape[1], 1))
        for _ in range(self.iterations):
            # Have to convert to numpy for exp as scipy cannot handle this
            self.w -= self.gamma * (X.T * csp.csr_matrix(Y / (1 + xp.exp((X * self.w).toarray()))))
        return self

    def predict(self, X):
        decision_function = X * self.w
        probabilities = 1 / (1 + xp.exp(-decision_function) - 1)
        labels = probabilities > 0.5
        return labels
