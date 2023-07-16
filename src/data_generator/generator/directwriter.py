import json
import os
from scipy.io import mmwrite

from data_generator.generator.mapping import Mapping


class DirectWriter:
    def __init__(self, mapping: Mapping, output_folder: str):

        # Validate output folder
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
        self.output_folder = output_folder
        self.mapping = mapping

    def write(self):
        self._write_target_table()
        for k in range(self.mapping.K):
            self._write_source_table(k)
            self._write_c_map(k)
            self._write_r_map(k)

    def _write_source_table(self, k):
        mmwrite(self.output_folder + "S{}".format(k + 1), self.mapping.data[k])

    def _write_target_table(self):
        mmwrite(self.output_folder + "T", self.mapping.T)

    def _write_c_map(self, k):
        if self.mapping.c_map != None:
            mmwrite(
                self.output_folder + "T_S{}_col_mapping".format(k + 1),
                self.mapping.c_map[k],
            )

    def _write_r_map(self, k):
        mmwrite(
            self.output_folder + "T_S{}_row_mapping".format(k + 1),
            self.mapping.r_map[k],
        )
