import json
import os

from data_generator.generator.mapping import Mapping


class Writer:
    def __init__(self, mapping: Mapping, output_folder: str, clear=False):

        # Validate output folder
        if os.path.isdir(output_folder):
            print(output_folder)
            if not os.listdir(output_folder):
                # If output folder is empty
                self.output_folder = output_folder
            else:
                # If output folder is not empty, ask to clear
                print("Output folder in home directory not empty.")
                if clear:
                    answer = "y"
                else:
                    answer = input(
                        "Do you want to clear the existing output folder %s? (y/n)"
                        % output_folder
                    )
                if answer == "n":
                    output_folder = output_folder[:-2] + "_new"
                    try:
                        os.mkdir(output_folder)
                    except OSError:
                        raise OSError(
                            "Creation of output folder %s failed" % output_folder
                        )
                if answer == "y":
                    try:
                        for f in os.listdir(output_folder):
                            os.remove(os.path.join(output_folder, f))
                    except OSError:
                        raise OSError(
                            "Clearing of output folder %s failed" % output_folder
                        )
                else:
                    raise ValueError("Invalid input.")
                print(
                    "A new output folder is created with the name %s!" % output_folder
                )
        else:
            try:
                os.mkdir(output_folder)
            except OSError:
                raise OSError("Creation of output folder %s failed" % output_folder)

        self.output_folder = output_folder
        self.mapping = mapping

    def write(self):
        self._write_target_table()
        for k in range(self.mapping.K):
            self._write_source_table(k)
            self._write_c_map(k)
            self._write_r_map(k)

    def _write_source_table(self, k):
        self.mapping.data[k].to_csv(self.output_folder + "S{}.csv".format(k + 1))

    def _write_target_table(self):
        self.mapping.T.to_csv(self.output_folder + "T.csv")

    def _write_c_map(self, k):
        if self.mapping.c_map != None:
            with open(
                self.output_folder + "T_S{}_col_mapping.json".format(k + 1),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.mapping.c_map[k], f, ensure_ascii=False, indent=4)

    def _write_r_map(self, k):
        with open(
            self.output_folder + "T_S{}_row_mapping.json".format(k + 1),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.mapping.r_map[k], f, ensure_ascii=False, indent=4)
