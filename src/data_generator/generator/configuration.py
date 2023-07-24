import configparser
from typing import Dict

import numpy as np


class Configuration:
    def __new__(self, cfgfile: str):
        cfg = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        cfg.read(cfgfile)
        self.output_folder = cfg["Paths"]["output_folder"]

        self.r_T = int(cfg["Properties"]["r_T"])
        self.c_T = int(cfg["Properties"]["c_T"])

        self.p = float(cfg["Properties"]["p"])

        self.join = cfg["Properties"]["join"]
        if self.join == "outer":
            self = self.outer(self, cfg)
        if self.join == "left":
            self = self.left(self, cfg)
        if self.join == "inner":
            self = self.inner(self, cfg)

        for k in range(self.n_R):
            if self.r_R[k] <= 0:
                raise ValueError('Table R_{} has invalid number of rows.'.format(k))
            if self.c_R[k] <= 0:
                raise ValueError('Table R_{} has invalid number of columns.'.format(k))
        if self.r_S <= 0:
            raise ValueError('Table S has invalid number of rows.')
        if self.c_S <= 0:
            raise ValueError('Table S has invalid number of columns.')

        return self

    def outer(self, cfg):
        rho_r_S = float(cfg["Properties"]["rho_r_S"])
        rho_c_S = float(cfg["Properties"]["rho_c_S"])
        rho_r_R = np.asarray(cfg["Properties"]["rho_r_R"].split(",")).astype(float)
        rho_c_R = np.asarray(cfg["Properties"]["rho_c_R"].split(",")).astype(float)
        t_T = float(cfg["Properties"]["t_T"])

        self.n_R = len(rho_c_R)
        self.n_S = 1

        # TODO: give warning when this does not return int
        self.r_S = np.round(self.r_T * rho_r_S).astype(int)
        self.c_S = np.round(self.c_T * rho_c_S).astype(int)
        self.r_R = np.round(self.r_T * rho_r_R).astype(int)
        self.c_R = np.round(self.c_T * rho_c_R).astype(int)

        self.n_target_matches = np.round(self.r_T * t_T).astype(int)
        self.n_col_matches = sum(self.c_R) + self.c_S - self.c_T
        # (assumption: we assume all r_R are the same)
        self.n_row_matches = self.r_R[0] - (
                (self.r_T - self.n_target_matches) - (self.r_S - self.n_target_matches)
        )
        # it is possible that we need matches between one R_k and S
        self.n_row_matches_r_s = (
                                         self.n_target_matches
                                         + self.n_row_matches
                                         + (self.r_S - self.n_target_matches - self.n_row_matches)
                                         + (self.r_R[0] - self.n_row_matches) * self.n_R
                                 ) - self.r_T

        return self

    def left(self, cfg):
        rho_c_S = float(cfg["Properties"]["rho_c_S"])
        rho_c_R = np.asarray(cfg["Properties"]["rho_c_R"].split(",")).astype(float)
        rho_r_R = np.asarray(cfg["Properties"]["rho_r_R"].split(",")).astype(float)
        t_T = float(cfg["Properties"]["t_T"])

        self.n_R = len(rho_c_R)
        self.n_S = 1

        # TODO: give warning when this does not return int
        self.r_S = self.r_T
        self.c_S = np.round(self.c_T * rho_c_S).astype(int)
        self.r_R = np.round(self.r_T * rho_r_R).astype(int)
        self.c_R = np.round(self.c_T * rho_c_R).astype(int)

        self.n_target_matches = np.round(self.r_T * t_T).astype(int)
        self.n_col_matches = sum(self.c_R) + self.c_S - self.c_T
        # (assumption: we assume all r_R are the same)
        self.n_row_matches = self.r_R[0]

        return self

    def inner(self, cfg):
        rho_c_S = float(cfg["Properties"]["rho_c_S"])
        rho_c_R = np.asarray(cfg["Properties"]["rho_c_R"].split(",")).astype(float)
        rho_r_R = np.asarray(cfg["Properties"]["rho_r_R"].split(",")).astype(float)

        self.n_R = len(rho_c_R)
        self.n_S = 1

        # TODO: give warning when this does not return int
        self.r_S = self.r_T
        self.c_S = np.round(self.c_T * rho_c_S).astype(int)
        self.r_R = np.round(self.r_T * rho_r_R).astype(int)
        self.c_R = np.round(self.c_T * rho_c_R).astype(int)

        self.n_target_matches = self.r_T - self.r_R[0]
        self.n_col_matches = sum(self.c_R) + self.c_S - self.c_T
        # (assumption: we assume all r_R are the same)
        self.n_row_matches = self.r_S - self.n_target_matches

        return self

    def to_dict(self) -> Dict[str, str]:
        """
        Converts all data chars in this config to a string representation and
        returns a dict of these.
        """
        return {
            "r_T": self.r_T,
            "c_T": self.c_T,
            "r_S": self.r_S,
            "c_S": self.c_S,
            "r_R": self.r_R,
            "c_R": self.c_R,
            "p_S_cfg": self.p,
            "n_row_matches": self.n_row_matches,
            "n_col_matches": self.n_col_matches,
            "n_target_matches": self.n_target_matches,
            "join": self.join,
            "n_S": self.n_S,
            "n_R": self.n_R,
        }
