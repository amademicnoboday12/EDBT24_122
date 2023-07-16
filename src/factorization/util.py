from __future__ import annotations

import glob
import json
import os
import statistics
from dataclasses import dataclass, asdict
from typing import List

import pandas as pd
from random import sample

from factorization.ilargi.IlargiData import IlargiData
from factorization.ilargi.IlargiMatrix import IlargiMatrix, csp, xp
from experiments.dataloader.datasets.loader import load_dataset
import numpy as np

class xpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, xp.integer):
            return int(obj)
        if isinstance(obj, xp.floating):
            return float(obj)
        if isinstance(obj, (np.ndarray, xp.ndarray)):
            return obj.tolist()
        return super(xpEncoder, self).default(obj)


@dataclass
class TimingResult:
    dataset: str
    join: str
    operator: str
    model: str
    tuple_ratio: float
    feature_ratio: float
    cardinality_T: int
    cardinality_S: int
    complexity: int
    selectivity: int
    times: List[float]
    num_cores: int = -1
    data_characteristics: dict = None
    profile_times: dict | None = None
    times_std: float | None = None
    times_mean: float | None = None

    def __str__(self):
        return json.dumps(asdict(self), cls=xpEncoder)

    def __post_init__(self):
        self.times_std = statistics.stdev(self.times) if len(self.times) > 1 else -1
        self.times_mean = statistics.mean(self.times)

@dataclass
class FeatureResult:
    dataset: str
    join: str
    features: List[float]
    r_S: List[int]
    c_S: List[int]
    nnz_S: List[int]
    r_T: int
    c_T: int
    nnz_T: int
    TR: float
    FR: float

    def __str__(self):
        return json.dumps(asdict(self), cls=xpEncoder)


def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = xp.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = xp.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = xp.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = xp.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


def load_and_preprocess(dataset: str, join: str, data_folder: str = None):
    NM: IlargiMatrix = load_dataset(dataset, data_folder)

    S_idxs = []
    if join == 'inner':
        S_idxs = []
        if dataset not in ['book', 'lastfm', 'movie', 'yelp']:
            NM.I[0] = None
    elif join == 'inner' or join == 'outer':
        if dataset in ['walmart', 'book', 'lastfm', 'movie', 'yelp']:
            S_idxs = [1]
        elif dataset in ['expedia', 'flight']:
            S_idxs = [2]
    if join == 'outer':
        if dataset in ['book', 'lastfm', 'movie', 'yelp']:
            S_idxs += [0]
    if join == 'union':
        if dataset not in ['book', 'lastfm', 'movie', 'yelp']:
            S_idxs = []
            for file in glob.glob('data/Hamlet/RealWorldDatasets/{}/S_*.csv'.format(dataset)):
                original_S = pd.read_csv(file)
            split_columns = original_S.iloc[:, 0].str.get(1)
            S = []
            I = []
            for value in split_columns.unique():
                locations = xp.where(split_columns == value)[0].tolist()
                print(locations)
                print(NM.S[0].shape)
                S.append(NM.S[0][locations, :])
                print(NM.I[0].shape)
                I.append(NM.I[0][:, locations])
            NM.S = S + NM.S[1:]
            NM.I = I + NM.I[1:]

    for S_idx in S_idxs:
        n_removed_rows = int(NM.S[S_idx].shape[0] * 0.2)
        removed_rows = sample(list(range(NM.S[S_idx].shape[0])), n_removed_rows)
        NM.S[S_idx] = delete_from_csr(NM.S[S_idx], removed_rows)
        NM.I[S_idx] = delete_from_csr(NM.I[S_idx], [], removed_rows)

    NM.M = None

    for k in range(NM.K):
        NM.S[k] = NM.S[k].tocsr()
        if NM.I[k] is not None:
            NM.I[k] = NM.I[k].tocsr()
    return NM


if __name__ == '__main__':
    load_and_preprocess('flight', 'inner')


def create_ilargi_matrix(dataset: str, join: str, dataset_type: str, dataset_folder: str = None) -> IlargiMatrix:
    # TODO calculate all data characteristics?
    if dataset_type == HAMLET:
        ilargi_matrix = load_and_preprocess(dataset, join, dataset_folder)
    elif dataset_type == GENERATED:
        ilargi_matrix = load_generated(directory=dataset)
    else:
        raise ValueError(f"invalid dataset_type: {dataset_type}")

    return IlargiMatrix(ilargi_matrix.S, ilargi_matrix.I, ilargi_matrix.M)


HAMLET = 'hamlet'
GENERATED = 'generated'


def aggregate_ncu_profile(profile_file: os.PathLike, n_repeats=None) -> pd.DataFrame:
    n_repeats = n_repeats or os.environ.get('NUM_REPEATS')
    df = pd.read_csv(profile_file, comment='=', header=0, usecols=[0, 4, 9, 11]) \
        .pivot(index=['ID', 'Kernel Name'], columns='Metric Name', values='Metric Value') \
        .reset_index(level=[0, 1])
    kernels_per_rep = len(df) // n_repeats
    df['repetition'] = pd.Series(df.index).apply(lambda id: int(id // kernels_per_rep + 1))
    return df.groupby(['repetition', 'Kernel Name']).agg(bytes_read_sum=("dram__bytes_read.sum","sum"), bytes_write_sum=("dram__bytes_write.sum", "sum"), times_called=("dram__bytes_write.sum", "count"))

def load_generated(directory: str) -> IlargiMatrix:
    AD = IlargiData(directory, direct=True)
    AM = IlargiMatrix(AD.S, AD.I, AD.M)
    return AM
