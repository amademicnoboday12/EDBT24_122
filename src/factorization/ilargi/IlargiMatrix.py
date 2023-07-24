import os

import numpy as np

from copy import copy

import scipy.sparse._csr
from loguru import logger

### CONFIGURATION

USE_MKL = os.environ.get("USE_MKL", "True") == "True"
if USE_MKL:
    try:
        import sparse_dot_mkl
    except ImportError:
        logger.exception(f"USE_MKL set to True, but could not import sparse_dot_mkl. Falling back to USE_MKL=False")
        USE_MKL = False

# Set array module (left from using GPU)
xp = np
csp = scipy.sparse


def dot_product(a, b, cast=True, **kwargs):
    if USE_MKL:
        return sparse_dot_mkl.dot_product_mkl(a, b, cast=cast, **kwargs)
    else:
        return a.dot(b)


def to_csr(X):
    if isinstance(X, dict):
        X = {key: csp.csr_matrix(arr, dtype=np.float64) if arr is not None else None for key, arr in
             X.items()} if X else X
    elif isinstance(X, list):
        X = [csp.csr_matrix(arr, dtype=np.float64) if arr is not None else None for arr in X]
    return X

def to_numpy_or_cupy(S, I, M):
    return S, I, M

class IlargiMatrix:
    """
    Store Ilargi Matrix and perform operations.
    """

    def __init__(self, S, I, M, shape=None, transpose=False):

        if M != None and shape is None:
            shape = (I[1].shape[0], M[1].shape[0])
        elif M is None and shape is None:
            shape = (I[1].shape[0], sum([S[k].shape[1] for k in range(len(S))]))

        xp = np
        self.shape = shape
        # Convert to CuPy if necessary.
        S, I, M = to_numpy_or_cupy(S, I, M)

        self.S = S
        self.I = I
        self.M = M

        self.transpose = transpose
        self.K = len(S)

        # return self

    def __str__(self):
        if self.I[0] is None:
            I = str(['No I[0]'] + [self.I[k].toarray() for k in range(1, self.K)])
        else:
            I = str([self.I[k].toarray() for k in range(self.K)])
        if self.M is None:
            M = 'No M'
        else:
            M = str([self.M[k].toarray() for k in range(self.K)])
        return "\n".join(
            [
                "S:", str([self.S[k].toarray() for k in range(self.K)]),
                "I:", I,
                "M:", M,
                "Shape:",
                str(self.shape),
            ]
        )

    def __repr__(self):
        return "\n".join(
            [
                "S:",
                repr(self.S),
                "I:",
                repr(self.I),
                "M:",
                repr(self.M),
                "Shape:",
                repr(self.shape),
            ]
        )

    def materialize(self):
        if self.M is None:
            if self.I[0] is None:
                result = self.S[0]
            else:
                result = self.I[0] * self.S[0]
            for k in range(1, self.K):
                if (csp.issparse(self.I[k]) and csp.issparse(self.S[k])):
                    p = dot_product(self.I[k], self.S[k], out=None, cast=True)
                else:
                    p = self.I[k] @ self.S[k]
                result = csp.hstack([result, p])
        else:
            if self.I[0] is None:
                result = self.S[0] * self.M[0].T
            else:
                result = multi_dot([self.I[0], self.S[0], self.M[0].T])
            for k in range(1, self.K):
                result += multi_dot([self.I[k], self.S[k], self.M[k].T])
        return result

    def random_choice(self):
        """
        Sample a random materialized row.
        """
        # Get index of random row from T
        rows = range(self.shape[0])
        random_row_idx = xp.random.choice(rows)
        # Initialize result with zeros
        random_row = xp.zeros(self.shape[1])
        raise NotImplementedError
        return random_row

    def _copy(self, S=[], I=[], M=[]):
        """
        Copy constructor
        """
        if S == []:
            S = self.S
        if I == []:
            I = self.I
        if M == []:
            M = self.M
        return IlargiMatrix(
            S=copy(S),
            I=copy(I),
            M=copy(M),
            shape=copy(self.shape),
            transpose=copy(self.transpose),
        )

    def __array_prepare__(self, obj, context=None):
        pass

    def __array_wrap__(self, out_arr, context=None):
        pass

    def __array_finalize__(self, obj):
        pass

    _SUPPORTED_UFUNCS = {
        xp.add: {1: "__add__", -1: "__radd__"},
        xp.subtract: {1: "__sub__", -1: "__rsub__"},
        xp.divide: {1: "__div__", -1: "__rdiv__"},
        xp.multiply: {1: "__mul__", -1: "__rmul__"},
        xp.power: {1: "__pow__", -1: "__rpow__"},
    }

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle ufunc supported in numpy standard library.

        :param ufunc: ufunc object
        :param method: type of method. In this class, only __call__ is handled
        :param inputs:
        :param kwargs:
        :return: Ilargi matrix or matrix or ndarray or numeric
        """
        if (
                ufunc in self._SUPPORTED_UFUNCS
                and len(inputs) == 2
                and method == "__call__"
        ):
            order = isinstance(inputs[0], IlargiMatrix) - isinstance(
                inputs[1], IlargiMatrix
            )
            if order == 1:
                return getattr(inputs[0], self._SUPPORTED_UFUNCS[ufunc][order])(
                    inputs[1], **kwargs
                )
            if order == -1:
                return getattr(inputs[1], self._SUPPORTED_UFUNCS[ufunc][order])(
                    inputs[0], **kwargs
                )
            if order == 0 and ufunc is xp.multiply:
                return inputs[0].__mul__(inputs[1], **kwargs)

        return NotImplemented

    # Element-wise Scalar Operators

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, IlargiMatrix):
            if self.shape == other.shape.T:
                return self._cross_prod()
            else:
                raise ValueError
        if csp.issparse(other):
            if self.transpose:
                return self._RMM(other.T).T
            else:
                return self._LMM(other)
        if isinstance(other, xp.ndarray):
            if self.transpose:
                return self._RMM(other.T).T
            else:
                return self._LMM(other)
        if xp.isscalar(other):
            # Generate a copy to return as result
            result = self._copy(self.S)
            # Perform multiplication on Si in result
            for i in range(self.K):
                result.S[i] = result.S[i] * other
            return result
        raise NotImplementedError

    def __rmul__(self, other):
        if csp.issparse(other):
            if self.transpose:
                return self._LMM(other.T).T
            else:
                return self._RMM(other)
        if xp.isscalar(other):
            # Generate a copy to return as result
            result = self._copy(self.S)
            # Perform multiplication on Si in result
            for i in range(self.K):
                result.S[i] = result.S[i] * other
            return result
        raise NotImplementedError

    def __div__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        # Generate a copy to return as result
        result = self._copy(self.S)
        # Perform multiplication on Si in result
        for i in range(self.K):
            result.S[i] = result.S[i].power(other)
        return result

    def __rpow__(self, other):
        raise NotImplementedError

    # Aggregation operators

    def power(self, other):
        return self.__pow__(other)

    def sum(self, axis=None):
        if axis is None:
            raise NotImplementedError
        elif axis == 0:
            return self._colSums()
        elif axis == 1:
            if self.transpose:
                return self._colSums()
            else:
                return self._rowSums()

    def _rowSums(self):
        if self.I[0] is None:
            result = self.S[0].sum(axis=1)
        else:
            result = self.I[0] * self.S[0].sum(axis=1)
        for k in range(1, self.K):
            result += self.I[k] * self.S[k].sum(axis=1)
        return result

    def _colSums(self):
        if self.M is None:
            if self.I[0] is None:
                result = self.S[0].sum(axis=0)
            else:
                result = self.I[0].sum(axis=0) * self.S[0]
            for k in range(1, self.K):
                result = xp.append(result, self.I[k].sum(axis=0) * self.S[k], axis=1)
        else:
            # TODO: use multidot
            if self.I[0] is None:
                result = self.S[0].sum(axis=0) * self.M[0].T
            else:
                result = self.I[0].sum(axis=0) * self.S[0] * self.M[0].T
            for k in range(1, self.K):
                result += self.I[k].sum(axis=0) * self.S[k] * self.M[k].T
        return result

    # Multiplication operators

    def dot(self, other):
        return self.__mul__(other)

    def _matmul_helper(self, arg):
        I, S, M, other, idx, counter = arg
        if (M is None):
            if (I[idx] is None):
                return dot_product(S[idx], other[counter:counter + S[idx].shape[1]], cast=True)
            else:
                return multi_dot([I[idx], S[idx], other[counter:counter + S[idx].shape[1]]])
        else:
            if (I[idx] is None):
                return multi_dot([S[idx], M[idx].T, other])
            else:
                return multi_dot([I[idx], S[idx], M[idx].T, other])

    def _LMM(self, other):
        counters = [0]
        counters.extend([self.S[i].shape[1] for i in range(0, self.K - 1)])
        counters = xp.cumsum(xp.array(counters))
        results = False
        # int() calls are needed because indexing a cupy ndarray returns a 0 dimensional ndarray.
        # By casting to an int we get the actual value.
        for i in range(self.K):
            if isinstance(results, bool):
                results = self._matmul_helper((self.I, self.S, self.M, other, i, int(counters[i])))
            else:
                results += self._matmul_helper((self.I, self.S, self.M, other, i, int(counters[i])))

        # with Pool(processes=max(self.K,5)) as pool:
        #     results=pool.map(self._matmul_helper,[(self.I, self.S, self.M, other, i, counters[i]) for i in range(self.K)])
        # r= xp.sum(results,axis=0)
        return results

    # def _LMM(self, other):

    #     print('entering LMM')
    #     start=time.time()
    #     if self.M is None:
    #         print("branch1.1")
    #         counter = self.S[0].shape[1]
    #         if self.I[0] is None:
    #             print("branch2.1")
    #             result = sparse_dot_mkl.dot_product_mkl(self.S[0],other[:counter],cast=True)
    #             # result = self.S[0].dot(other[:counter])
    #         else:
    #             print("branch2.2")
    #             result = multi_dot([self.I[0], self.S[0], other[:counter]])

    #         for k in range(1, self.K):
    #             print(f"for loop {k}")
    #             result += multi_dot(
    #                 [
    #                     self.I[k],
    #                     self.S[k],
    #                     other[counter : counter + self.S[k].shape[1]],
    #                 ]
    #             )
    #             counter += self.S[k].shape[1]
    #     else:
    #         if self.I[0] is None:
    #             result = multi_dot([self.S[0], self.M[0].T, other])
    #         else:
    #             result = multi_dot([self.I[0], self.S[0], self.M[0].T, other])
    #         for k in range(1, self.K):
    #             result += multi_dot([self.I[k], self.S[k], self.M[k].T, other])

    #     end=time.time()
    #     print('time cost',end-start,'s')
    #     return result

    def _RMM(self, other):
        if self.M is None:
            if self.I[0] is None:
                result = other * self.S[0]
            else:
                result = multi_dot([other, self.I[0], self.S[0]])
            for k in range(1, self.K):
                result = csp.hstack([result, multi_dot([other, self.I[k], self.S[k]])])
        else:
            if self.I[0] is None:
                result = multi_dot([other, self.S[0], self.M[0].T])
            else:
                result = multi_dot([other, self.I[0], self.S[0], self.M[0].T])
            for k in range(1, self.K):
                result += multi_dot([other, self.I[k], self.S[k], self.M[k].T])
        return result

    def _cross_prod(self):
        return NotImplementedError

    @property
    def T(self):
        return IlargiMatrix(self.S, self.I, self.M, self.shape, transpose=True)


# Adapted from https://github.com/numpy/numpy/blob/v1.23.0/numpy/linalg/linalg.py#L2617-L2735

def multi_dot(arrays, *, out=None):
    """
    Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.
    `multi_dot` chains `numpy.dot` and uses optimal parenthesization
    of the matrices [1]_ [2]_. Depending on the shapes of the matrices,
    this can speed up the multiplication a lot.
    """
    assert out is None  # scipy spmatrix dot product can't handle out so we need to make sure it is not used
    n = len(arrays)
    # optimization only makes sense for len(arrays) > 2
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return dot_product(arrays[0], arrays[1], out=out)
        # return sparse_dot_mkl.dot_product_mkl(arrays[0], arrays[1], out=out, cast=True)

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)

    return result


def _multi_dot_three(A, B, C, out=None):
    """
    Find the best order for three arrays and do the multiplication.
    For three arguments `_multi_dot_three` is approximately 15 times faster
    than `_multi_dot_matrix_chain_order`
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return dot_product(dot_product(A, B), C, out=out)
        # return sparse_dot_mkl.dot_product_mkl(sparse_dot_mkl.dot_product_mkl(A, B, cast=True), C, out=out, cast=True)
    else:
        return dot_product(A, dot_product(B, C), out=out)
        # return sparse_dot_mkl.dot_product_mkl(A, sparse_dot_mkl.dot_product_mkl(B, C, cast=True), out=out, cast=True)


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """
    Return a xp.array that encodes the optimal order of mutiplications.
    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.
    Also return the cost matrix if `return_costs` is `True`
    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.
        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])
    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = xp.zeros((n, n), dtype=xp.double)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = xp.empty((n, n), dtype=xp.intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = xp.Inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s


def _multi_dot(arrays, order, i, j, out=None):
    """Actually do the multiplication with the given order."""
    if i == j:
        # the initial call with non-None out should never get here
        assert out is None

        return arrays[i]
    else:
        a = _multi_dot(arrays, order, i, int(order[i, j]))
        b = _multi_dot(arrays, order, int(order[i, j]) + 1, j)
        return dot_product(a, b)
        # return sparse_dot_mkl.dot_product_mkl(_multi_dot(arrays, order, i, order[i, j]),
        #                                       _multi_dot(arrays, order, order[i, j] + 1, j),
        #                                       out=out, cast=True)
