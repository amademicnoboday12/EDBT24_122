import unittest

from factorization.ilargi.IlargiData import IlargiData
from factorization.ilargi.IlargiMatrix import IlargiMatrix, xp, csp

from factorization.ilargi.algorithms.LinearRegression import LinearRegression
from factorization.ilargi.algorithms.LogisticRegression import LogisticRegression
from factorization.ilargi.algorithms.GaussianNMF import GaussianNMF
from factorization.ilargi.algorithms.KMeans import KMeans


class TestIlargiMatrixOperations(unittest.TestCase):
    """
    Test whether the class IlargiMatrix correctly performs linear algebra operations.
    """

    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/all_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)

    def test_linear_regression(self):
        m = LinearRegression(0.1, 1)
        Y = csp.csr_matrix(xp.ones((self.T.shape[0], 1)))
        m.fit(m, self.T, Y)
        m.fit(m, self.AM, Y)

    def test_logistic_regression(self):
        m = LogisticRegression(0.1, 1)
        Y = csp.csr_matrix(xp.ones((self.T.shape[0], 1)))
        m.fit(m, self.T, Y)
        m.fit(m, self.AM, Y)

    def test_gaussian_nmf(self):
        m = GaussianNMF(2, 1)
        m.fit(m, self.T)
        m.fit(m, self.AM)

    def test_kmeans(self):
        m = KMeans(2, self.T, 5)
        m.fit(m, self.T)
        m = KMeans(2, self.AM, 5)
        m.fit(m, self.AM)


# Inner
## Two tables
class TestNoRedundancyInner(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/no_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestSourceRedundancyInner(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/source_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestTargetRedundancyInner(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/target_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestAllRedundancyInner(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/all_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


## Three tables


class TestNoRedundancyInnerThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/no_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestSourceRedundancyInnerThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/source_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestTargetRedundancyInnerThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/target_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestAllRedundancyInnerThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/inner/all_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


# Left
## Two tables


class TestNoRedundancyLeft(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/no_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestSourceRedundancyLeft(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/source_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestTargetRedundancyLeft(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/target_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestAllRedundancyLeft(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/all_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


## Three tables


class TestNoRedundancyLeftThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/no_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestSourceRedundancyLeftThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/source_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestTargetRedundancyLeftThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/target_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestAllRedundancyLeftThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/left/all_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


# Outer
## Two tables


class TestNoRedundancyOuter(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/no_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestSourceRedundancyOuter(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/source_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestTargetRedundancyOuter(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/target_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestAllRedundancyOuter(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/all_redundancy/"
        AD = IlargiData(folder, K=2, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


## Three tables


class TestNoRedundancyOuterThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/no_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestSourceRedundancyOuterThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/source_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestTargetRedundancyOuterThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/target_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


class TestAllRedundancyOuterThree(TestIlargiMatrixOperations):
    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/all_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)


if __name__ == "__main__":
    unittest.main()
