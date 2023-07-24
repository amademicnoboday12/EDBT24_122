import pandas as pd
import numpy as np
import itertools
from scipy.sparse import csr_matrix


class Mapping:
    def __new__(self, cfg, direct):

        # Initialize empty lists to store indexes for each R table
        R_r_idxs = {r: [] for r in range(cfg.n_R)}
        R_c_idxs = {r: [] for r in range(cfg.n_R)}

        # Initialize empty list to store indexes for S
        S_r_idxs = []
        S_c_idxs = []

        # Start rows indicate which rows have already been added to S and R
        start_r_R = {r: 0 for r in range(cfg.n_R)}
        start_c_R = {r: 0 for r in range(cfg.n_R)}
        start_r_S = 0
        start_c_S = 0

        ## Add source redundancy
        if cfg.n_col_matches > cfg.c_S:
            raise NotImplementedError(
                "Number of column matches cannot exceed number of columns in S."
            )
        if cfg.n_col_matches > 0:
            # For each source table
            # (assumption: a column is not present in more than two source tables)
            for k in range(cfg.n_R):
                # If the number of columns treated in the loop is smaller than the number of column matches
                if sum(cfg.c_R[:k]) < cfg.n_col_matches:
                    # Add column matches to indexes, but do not add more than the number of columns in k
                    R_c_idxs_k = list(
                        range(cfg.n_col_matches)[
                            sum(cfg.c_R[:k]) : sum(cfg.c_R[:k]) + cfg.c_R[k]
                        ]
                    )
                    R_c_idxs[k] += R_c_idxs_k
                start_c_R[k] += cfg.n_col_matches

            S_c_idxs += list(range(cfg.n_col_matches))
            start_c_S += cfg.n_col_matches

        ## Add target redundancy
        if cfg.n_target_matches > 0:
            # Create list of indices which will be shared between tables
            # (assumption: target matches always use index 0)
            # (assumption: each table has a row with index 0)
            target_matches_r_idxs = list([0] * (cfg.n_target_matches))

            # Add full list of indices to S
            S_r_idxs += target_matches_r_idxs

        ## Add row matches

        # For each source table
        # (assumption: a row match is between all tables)
        # (assumption: row matches reuse indexes ranging from 0 to the number of row matches)
        start_row = 0
        for k in range(cfg.n_R):
            R_r_idxs[k] += range(start_row, cfg.n_row_matches + start_row)
            start_r_R[k] += cfg.n_row_matches
        S_r_idxs += range(start_row, cfg.n_row_matches + start_row)
        start_r_S += cfg.n_row_matches

        # Special case of row matches for outer join
        if cfg.join == "outer":
            start_r_R_before_row_matches = start_r_R.copy()
            n_row_matches = [0] * cfg.n_R
            remaining_matches = cfg.n_row_matches_r_s
            for k in range(cfg.n_R):
                n_row_matches[k] = max(remaining_matches, cfg.r_R[k] - len(R_r_idxs[k]))
                remaining_matches = remaining_matches - n_row_matches[k]
                R_r_idxs[k] += range(start_r_R[k], n_row_matches[k] + start_r_R[k])
                for k2 in range(cfg.n_R):
                    start_r_R[k2] += n_row_matches[k]
                if remaining_matches == 0:
                    break
            S_r_idxs += range(start_r_S, cfg.n_row_matches_r_s + start_r_S)
            start_r_S += cfg.n_row_matches_r_s

        ## Add non-matching rows

        for k in range(cfg.n_R):
            n_new_rows = cfg.r_R[k] - len(R_r_idxs[k])
            R_r_idxs[k] += [start_r_R[k] + i for i in range(n_new_rows)]
            for k2 in range(cfg.n_R):
                start_r_R[k2] += n_new_rows
        S_r_idxs += [start_r_R[k] + i for i in range(cfg.r_S - len(S_r_idxs))]

        ## Add non-matching columns

        for k in range(cfg.n_R):
            n_new_cols = cfg.c_R[k] - len(R_c_idxs[k])
            R_c_idxs[k] += [start_c_R[k] + i for i in range(n_new_cols)]
            start_c_S += n_new_cols
            for k2 in range(cfg.n_R):
                start_c_R[k2] += n_new_cols
        S_c_idxs += [start_c_R[k] + i for i in range(cfg.c_S - len(S_c_idxs))]

        ## Create empty R
        # Initialize empty dict to store R dataframes
        self.R = [None] * cfg.n_R
        # Create a df for each R filled with markers
        for k in range(cfg.n_R):
            data = np.full((cfg.r_R[k], cfg.c_R[k]), np.inf)
            self.R[k] = pd.DataFrame(data, columns=R_c_idxs[k], index=R_r_idxs[k])

        ## Create a df for S filled with markers
        data = np.full((cfg.r_S, cfg.c_S), np.inf)
        self.S = pd.DataFrame(data, columns=S_c_idxs, index=S_r_idxs)

        ## Create T
        # Use indices to create T with correct row and column indices
        # We have to drop the column matches in R as pandas cannot handle these in the join
        if cfg.n_col_matches > 0:
            R = self.R.copy()
            for idx in range(cfg.n_R):
                for c_match in range(cfg.n_col_matches):
                    if c_match in R[idx].columns:
                        R[idx] = R[idx].drop(columns=[c_match])
                    else:
                        R[idx] = R[idx]
        else:
            R = self.R.copy()

        self.T = self.S.join(
            R,
            how=cfg.join,
        )

        # Special case for outer join, where we have to add back items which are in a
        # column match but which are not a row match with S
        if cfg.join == "outer":
            remaining_row_matches = cfg.n_row_matches_r_s
            for k in range(cfg.n_R):
                remaining_row_matches = remaining_row_matches - n_row_matches[k]
                r_idxs = list(
                    range(
                        start_r_R_before_row_matches[k],
                        n_row_matches[k] + start_r_R_before_row_matches[k],
                    )
                )
                for r_idx in r_idxs:
                    self.T.iloc[
                        self.T.index.get_loc(r_idx), : cfg.n_col_matches
                    ] = np.inf
                for k in range(cfg.n_R):
                    start_r_R_before_row_matches[k] += n_row_matches[k]
                if remaining_row_matches == 0:
                    break

        T_as_numpy = self.T.to_numpy()

        # Add random data to T where T is not null
        not_null = np.argwhere(np.isinf(T_as_numpy)).T
        not_null_rows = not_null[0]
        not_null_cols = not_null[1]

        # Add normal data
        random_data = np.random.randint(1, 10, size=len(not_null_rows))

        T_as_numpy[not_null_rows, not_null_cols] = random_data

         # Add sparsity by adding zeros
        n_zeroes_cols = int(cfg.p * cfg.c_S)
        T_as_numpy[:cfg.r_S, :n_zeroes_cols] = 0

        self.T[:] = T_as_numpy

        # Add target redundancy elements in T
        if cfg.n_target_matches > 0:
            rows = list(set(self.T.index[: cfg.n_target_matches + 1]))
            cols = list(
                itertools.chain.from_iterable(
                    [list(self.R[k].columns) for k in range(cfg.n_R)]
                )
            )
            target_matches = self.T.loc[rows, cols]

            self.T.loc[rows, cols] = np.full(target_matches.shape, 4)

        self.T = self.T.astype('float') # int cannot handle Null values

        ## Add data to R and create mappings
        self.K = cfg.n_R + cfg.n_S
        R_r_map = {r: [] for r in range(cfg.n_R)}
        R_c_map = {r: [] for r in range(cfg.n_R)}
        S_r_map = []
        S_c_map = []

        # Add data to R
        for k in range(cfg.n_R):
            R = self.R[k].to_numpy()
            for i, r_idx in enumerate(R_r_idxs[k]):
                row_idxs = self.R[k].index.get_loc(r_idx)
                for j, c_idx in enumerate(R_c_idxs[k]):
                    col_idxs = self.R[k].columns.get_loc(c_idx)
                    element = self.T.loc[r_idx, c_idx]
                    single = isinstance(element, float)
                    if single:
                        R[row_idxs, col_idxs] = element
                    # If there are multiple elements, they are all the same so we pick the first one
                    else:
                        R[row_idxs, col_idxs] = list(element)[k]
                # Row mapping
                if single:
                    R_r_map[k] += [(i, self.T.index.get_loc(r_idx))]
                # If there are multiple elements, map our row to all of them
                else:
                    R_r_map[k] += [
                        (i, idx)
                        for idx in range(self.T.shape[0])[self.T.index.get_loc(r_idx)]
                    ]
            self.R[k][:] = R.astype(int)
            # Column mapping
            for j, c_idx in enumerate(R_c_idxs[k]):
                R_c_map[k] += [(j, self.T.columns.get_loc(c_idx))]

        ## Add data to S and create mappings
        # First treat target matches
        S = self.S.to_numpy()
        for i in range(cfg.n_target_matches + 1):
            S_r_map += [(i, i)]
            for j, c_idx in enumerate(S_c_idxs):
                # If there are multiple rows with the same index, add all of them
                S[i, j] = self.T.iloc[i].loc[c_idx]
        
        for i, r_idx in enumerate(S_r_idxs[cfg.n_target_matches + 1 :]):
            i = i + cfg.n_target_matches + 1
            S_r_map += [(i, cfg.n_target_matches + r_idx)]
            for j, c_idx in enumerate(S_c_idxs):
                # If there are multiple rows with the same index, add all of them
                S[i, j] = self.T.loc[r_idx, c_idx]
        for j, c_idx in enumerate(S_c_idxs):
            S_c_map += [(j, self.T.columns.get_loc(c_idx))]

        self.S[:] = S.astype(int)

        # Turn NaNs into zeroes
        self.T[np.isnan(self.T)] = 0.0

        if direct:

            ## Refine row and col matches

            self.r_map = {}
        
            self.S = csr_matrix(self.S.to_numpy())
            self.r_map[0] = self.mapping(S_r_map, (self.T.shape[0], self.S.shape[0]))

            for k in range(cfg.n_R):
                self.R[k] = csr_matrix(self.R[k].to_numpy())
                self.r_map[k + 1] = self.mapping(
                    R_r_map[k], (self.T.shape[0], self.R[k].shape[0])
                )
                
            if cfg.n_col_matches > 0:
                self.c_map = {}
                self.c_map[0] = self.mapping(S_c_map, (self.T.shape[1], self.S.shape[1]))
                for k in range(cfg.n_R):
                    self.c_map[k + 1] = self.mapping(
                        R_c_map[k], (self.T.shape[1], self.R[k].shape[1])
                    )
            else:
                self.c_map = None

            self.T = csr_matrix(self.T.to_numpy())

        else:

            ## Refine row mappings

            self.r_map = {}

            # Refine row mappings for S
            self.r_map[0] = {"join_type": cfg.join, "row_mapping": []}
            for row_s, row_t in S_r_map:
                entry = {
                    "s": 0,
                    "row_s": int(row_s),
                    "row_t": int(row_t),
                }
                self.r_map[0]["row_mapping"].append(entry)

            # Refine row mappings for R
            for k in range(cfg.n_R):
                self.r_map[k + 1] = {"join_type": cfg.join, "row_mapping": []}
                for row_s, row_t in R_r_map[k]:
                    entry = {
                        "s": k + 1,
                        "row_s": int(row_s),
                        "row_t": int(row_t),
                    }
                    self.r_map[k + 1]["row_mapping"].append(entry)

            ## Refine col mappings

            if cfg.n_col_matches == 0:
                self.c_map = None

            else:

                self.c_map = {}

                # Refine row mappings for S
                self.c_map[0] = {"join_type": cfg.join, "col_mapping": []}
                for col_s, col_t in S_c_map:
                    entry = {
                        "s": 0,
                        "col_s": int(col_s),
                        "col_t": int(col_t),
                    }
                    self.c_map[0]["col_mapping"].append(entry)

                # Refine row mappings for R
                for k in range(cfg.n_R):
                    self.c_map[k + 1] = {"join_type": cfg.join, "col_mapping": []}
                    for col_s, col_t in R_c_map[k]:
                        entry = {
                            "s": k + 1,
                            "col_s": int(col_s),
                            "col_t": int(col_t),
                        }
                        self.c_map[k + 1]["col_mapping"].append(entry)

        ## Store remaining information
        self.data = [*[self.S], *self.R]

        # self.check(self, cfg)

        return self

    def check(self, cfg):
        # Pandas does not implement outer join in the same way
        # TODO: implement check for outer join
        if cfg.join != "outer":
            if cfg.n_col_matches > 0:
                if cfg.n_col_matches > 0:
                    R = self.R.copy()
                    for idx in range(cfg.n_R):
                        for c_match in range(cfg.n_col_matches):
                            if c_match in R[idx].columns:
                                R[idx] = R[idx].drop(columns=[c_match])
                            else:
                                R[idx] = R[idx]
                else:
                    R = self.R.copy()
            else:
                R = self.R
            join = self.S.join(R, how=cfg.join)
            if not join.equals(self.T):
                raise AssertionError(
                    "Join incorrect.\nJoin:\n{}\nT:\n{}\nData:\n{}".format(
                        join, self.T, self.data
                    )
                )

        if not self.T.shape == (cfg.r_T, cfg.c_T):
            raise AssertionError("Shape of T incorrect.\nT:{}".format(self.T))

    def mapping(mapping, shape):
        row_array, col_array = [], []
        for (dim_s, dim_t) in mapping:
            row_array.append(dim_t)
            col_array.append(dim_s)
        mapping = csr_matrix(
            (np.ones(len(row_array)), (row_array, col_array)),
            shape=shape,
            dtype=np.int32,
        )
        return mapping
