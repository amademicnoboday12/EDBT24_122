import os

import unittest

from factorization.ilargi.IlargiData import IlargiData
from factorization.ilargi.IlargiMatrix import IlargiMatrix, xp, csp

class TestIlargiMatrixOperations(unittest.TestCase):
    """
    Test whether the class ilargiMatrix correctly performs linear algebra operations.
    """

    @classmethod
    def setUpClass(self):
        folder = "tests/resources/generated/data/outer/all_redundancy_three/"
        AD = IlargiData(folder, K=3, direct=True)
        self.T = AD.T
        self.AM = IlargiMatrix(AD.S, AD.I, AD.M, shape=self.T.shape)

    def test_materialize(self):
        baseline = self.T
        ilargi = self.AM.materialize()
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))

    def test_multiply(self):
        x = 3
        baseline = self.T * x
        ilargi = (self.AM * x).materialize()
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))

        baseline = x * self.T
        ilargi = (x * self.AM).materialize()
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))

    def test_power(self):
        x = 2
        baseline = self.T.power(x)
        ilargi = (self.AM**x).materialize()
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))

    def test_rowSums(self):
        baseline = self.T.sum(axis=1)
        ilargi = self.AM.sum(axis=1)
        self.assertTrue(xp.allclose(baseline, ilargi))

    def test_colSums(self):
        baseline = self.T.sum(axis=0)
        ilargi = self.AM.sum(axis=0)
        self.assertTrue(xp.allclose(baseline, ilargi))

    def test_LMM(self):
        X = csp.csr_matrix(xp.ones((self.AM.shape[1], 2), dtype=xp.float32))
        baseline = self.T.dot(X)
        ilargi = self.AM * X
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))

    def test_LMM_transpose(self):
        X = csp.csr_matrix(xp.ones((self.AM.shape[0], 2)))
        baseline = self.T.T.dot(X)
        ilargi = self.AM.T.dot(X)
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))

    def test_RMM(self):
        X = csp.csr_matrix(xp.ones((2, self.AM.shape[0])))
        baseline = X.dot(self.T)
        ilargi = self.AM.__rmul__(X)
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))

    def test_RMM_transpose(self):
        X = csp.csr_matrix(xp.ones((2, self.AM.shape[1])))
        baseline = X.dot(self.T.T)
        ilargi = self.AM.T.__rmul__(X)
        # TODO
        self.assertTrue(xp.allclose(baseline.toarray(), ilargi.toarray()))



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
