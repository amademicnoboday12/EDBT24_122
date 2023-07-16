import numpy as np
from scipy.io import mmread
from scipy.sparse import hstack
from factorization.ilargi.IlargiMatrix import IlargiMatrix
from scipy.sparse import csr_matrix


def load_expedia(base_dir: str = "data/Hamlet/Expedia"):
    S = mmread(f"{base_dir}/MLSSparse.txt").tocsr()

    join_set1 = (
        np.genfromtxt(f"{base_dir}/MLFK1.csv", skip_header=True, dtype=int) - 1
    )
    R1 = mmread(
        f"{base_dir}/MLR1Sparse.txt",
    ).tocsr()

    join_set2 = (
        np.genfromtxt(f"{base_dir}/MLFK2.csv", skip_header=True, dtype=int) - 1
    )
    R2 = mmread(
        f"{base_dir}/MLR2Sparse.txt",
    ).tocsr()

    Y = np.matrix(
        np.genfromtxt(f"{base_dir}/MLY.csv", skip_header=True, dtype=int)
    ).T

    T = hstack((S, R1[join_set1], R2[join_set2]))

    rows, columns, data = [], [], []
    for i, row in enumerate(range(S.shape[0])):
        rows += [i]
        columns += [row]
        data += [1]
    I0 = csr_matrix((data, (rows, columns)), (T.shape[0], S.shape[0]))

    rows, columns, data = [], [], []
    for i, row in enumerate(join_set1.tolist()):
        rows += [i]
        columns += [row]
        data += [1]
    I1 = csr_matrix((data, (rows, columns)), (T.shape[0], R1.shape[0]))

    rows, columns, data = [], [], []
    for i, row in enumerate(join_set2.tolist()):
        rows += [i]
        columns += [row]
        data += [1]
    I2 = csr_matrix((data, (rows, columns)), (T.shape[0], R2.shape[0]))

    I = [I0, I1, I2]

    rows, columns, data = [], [], []
    for i, col in enumerate(range(S.shape[1])):
        rows += [col]
        columns += [i]
        data += [1]
    M0 = csr_matrix((data, (rows, columns)), (T.shape[1], S.shape[1]))

    rows, columns, data = [], [], []
    for i, col in enumerate(range(R1.shape[1])):
        rows += [col]
        columns += [i]
        data += [1]
    M1 = csr_matrix((data, (rows, columns)), (T.shape[1], R1.shape[1]))

    rows, columns, data = [], [], []
    for i, col in enumerate(range(R1.shape[1], R1.shape[1] + R2.shape[1])):
        rows += [col]
        columns += [i]
        data += [1]
    M2 = csr_matrix((data, (rows, columns)), (T.shape[1], R2.shape[1]))

    M = [M0, M1, M2]

    NM = IlargiMatrix([S, R1, R2], I, M)
    return NM