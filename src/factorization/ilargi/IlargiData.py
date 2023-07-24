import pandas as pd
import numpy as np
import json
import os
from scipy.sparse import csr_matrix, issparse
from scipy.io import mmread
from factorization.ilargi.IlargiMatrix import csp


class IlargiData:
    """
    Parses a folder and returns the input to an IlargiMatrix object.
    """

    def __new__(self, folder, K=None, sep=",", direct=False):
        # TODO infer K
        self.K = K or 2
        # Obtain data paths
        T_data_path = folder + "T"
        Y_data_path = folder + "Y"
        self.S_data_paths = {}
        self.S_col_mapping_paths = {}
        self.S_row_mapping_paths = {}
        for k in range(self.K):
            # Data path source tables
            self.S_data_paths[k] = folder + "S{}".format(k + 1)
            # Data path col mapping
            self.S_col_mapping_paths[k] = folder + "T_S{}_col_mapping".format(k + 1)
            # Data path row mapping
            self.S_row_mapping_paths[k] = folder + "T_S{}_row_mapping".format(k + 1)
        if direct:
            self.T = mmread(T_data_path + ".mtx")
            if (issparse(self.T)):
                self.T = self.T.tocsr()
            self.S = [None] * self.K
            self.I = {}
            self.M = {}
            for k in range(self.K):
                self.S[k] = mmread(self.S_data_paths[k] + ".mtx")
                if (issparse(self.S[k])):
                    self.S[k] = self.S[k].tocoo()
                if os.path.isfile(self.S_col_mapping_paths[k] + ".mtx"):
                    self.M[k] = mmread(self.S_col_mapping_paths[k] + ".mtx")
                    if (issparse(self.M[k])):
                        self.M[k] = self.M[k].tocoo()
                else:
                    self.M = None
                self.I[k] = mmread(self.S_row_mapping_paths[k] + ".mtx")
                if (issparse(self.I[k])):
                    self.I[k] = self.I[k].tocoo()
        else:
            # Get tables T and Y
            self.T, self.T_cols = self._parse_T(self, T_data_path + ".csv", sep=sep)
            if os.path.isfile(Y_data_path):
                self.Y = self._parse_Y(self, Y_data_path + ".csv", sep=sep)

            # Obtain S and mappings
            self.S, self.S_cols = self._parse_S(self, self.S_data_paths, sep)

            # Obtain compressed indicator and mapping matrices
            self.I = self._mapping(self, self.S_row_mapping_paths, dim=0)
            if os.path.isfile(self.S_col_mapping_paths[0] + ".json"):
                self.M = self._mapping(self, self.S_col_mapping_paths, dim=1)
            else:
                self.M = None

        # TODO: I[0] not necessary for source redundancy removal (not hard to implement though)
        if self.M != None:
            # Remove source redundancy from S
            self.S = self._remove_redundancy(self)

        # Morpheus-like equations
        if self.S[k].shape[0] == self.T.shape[0]:
            self.I[0] = None

        for k in range(self.K):
            self.S[k] = csp.csr_matrix(self.S[k], dtype='float64')
            if (self.I is not None) and (self.I[k] is not None):
                self.I[k] = csp.csr_matrix(self.I[k], dtype='float64')
            if (self.M is not None) and (self.M[k] is not None):
                self.M[k] = csp.csr_matrix(self.M[k], dtype='float64')
        self.T = csp.csr_matrix(self.T)
        return self

    def _parse_T(self, data_path, sep=","):
        """
        Parses the target table.
        """
        # load data
        T = pd.read_csv(data_path, sep=sep, index_col=0)
        # save source columns
        T_cols = list(T.columns)
        # convert to sparse csr format
        T = csr_matrix(T.to_numpy())
        return T, T_cols

    def _parse_Y(self, data_path, sep=","):
        """
        Parses the target table.
        """
        # load data
        Y = pd.read_csv(data_path, sep=sep)
        # convert to numpy array
        # TODO [:, -1] is a hacky way of dealing with cases where some data have index and some do not
        Y = Y.to_numpy()[:, -1]
        return Y

    def _parse_S(self, S_data_paths, sep=";", to_df=False):
        """
        Parses all source tables.
        """
        S = [None] * self.K
        S_cols = [None] * self.K
        for s1 in range(self.K):
            S[s1], S_cols[s1] = self._parse_s(
                self, S_data_paths[s1] + ".csv", sep=sep, to_df=to_df
            )
        return S, S_cols

    def _parse_s(self, data_path, sep=";", to_df=False):
        """
        Parses a single source table.
        """
        # load data
        S = pd.read_csv(data_path, sep=sep, index_col=0)
        # save source columns
        S_cols = list(S.columns)
        # convert to sparse coo format
        if not to_df:
            S = csr_matrix(S.to_numpy().astype(float))

        return S, S_cols

    def S_as_df(self):
        """
        Returns S as list of dfs for testing purposes
        """
        S, _ = self._parse_S(self, self.S_data_paths, to_df=True)
        return S

    def _mapping(self, json_file_paths, dim):
        """
        Returns the row mappings as a dictionary
        """
        if dim == 0:
            dim_str = "row"
        elif dim == 1:
            dim_str = "col"
        else:
            raise ValueError
        mappings = {}
        for _, json_file_path in json_file_paths.items():
            with open(json_file_path + ".json", "r") as f:
                mapping = json.load(f)["{}_mapping".format(dim_str)]
            k = mapping[0]["s"]
            indicator_shape = (self.T.shape[dim], self.S[k].shape[dim])
            row_array, col_array = [], []
            for match in mapping:
                dim_s = match["{}_s".format(dim_str)]
                dim_t = match["{}_t".format(dim_str)]
                row_array.append(dim_t)
                col_array.append(dim_s)
            mappings[k] = csr_matrix(
                (np.ones(len(row_array)), (row_array, col_array)),
                shape=indicator_shape,
                dtype=np.int32,
            )
        return mappings

    def _remove_redundancy(self):
        # TODO: more efficient method
        for k in range(self.K):
            if k != 0:
                # Convert to lil format because we cannot set items in coo matrices
                self.S[k] = self.S[k].tolil()
                # Use tables 0:(n-1) as base table for generating redundancy matrix n
                base_tables = range(0, k)
                rows_k = list(self.I[k].row)
                cols_k = list(self.M[k].row)
                for base_table in base_tables:
                    rows_base_table = list(self.I[base_table].row)
                    cols_base_table = list(self.M[base_table].row)
                    redundant_rows, ind_row_base, ind_row_target = np.intersect1d(
                        rows_base_table, rows_k, return_indices=True
                    )
                    redundant_cols, ind_col_base, ind_col_target = np.intersect1d(
                        cols_base_table, cols_k, return_indices=True
                    )
                    for rid, row in enumerate(redundant_rows):
                        for cid, col in enumerate(redundant_cols):
                            # Store indexes of redundant locations
                            # TODO: switch to new I/M format
                            S_row = self.I[k].col[ind_row_target[rid]]
                            S_col = self.M[k].col[[ind_col_target[cid]]]
                            self.S[k][S_row, S_col] = 0
                # Covert back to csr format
                self.S[k] = self.S[k].tocsr()
        return self.S
