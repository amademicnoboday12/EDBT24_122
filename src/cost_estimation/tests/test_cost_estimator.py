import unittest

from factorization.ilargi.IlargiData import IlargiData
from factorization.ilargi.IlargiMatrix import IlargiMatrix
from estimator.CostEstimator import CostEstimator


class TestCostEstimator(unittest.TestCase):
    """
    Test whether the class IlargiMatrix correctly performs linear algebra operations.
    """

    @classmethod
    def setUpClass(self):
        folder = "data/Generator/thesis_examples/outer/all_redundancy/"
        AD = IlargiData(folder, K=2)
        AM = IlargiMatrix(AD.S, AD.I, AD.M)
        self.CE = CostEstimator(AM)

    def test(self):
        print("")


if __name__ == "__main__":
    unittest.main()
