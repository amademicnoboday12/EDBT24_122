import configparser
import pandas as pd

from .configuration import Configuration
from .mapping import Mapping
from .writer import Writer
from .directwriter import DirectWriter


class Generator:
    def __init__(self, cfgfile):
        self.config = Configuration(cfgfile)

    def read_config(self, cfgfile: str):
        cfg = configparser.ConfigParser()
        cfg.read(cfgfile)

    def read_data(self, infile: str, key: str):
        df = pd.read_csv(infile, delimiter=",", index_col=key)
        # fill in null values
        df = df.fillna(0)
        # convert to numerical
        df = self.convert_objects(df)
        return df

    def convert_objects(self, df):
        """
        Turn non-numerical columns into numerical
        """
        obj_cols = df.select_dtypes(["object"]).columns
        df[obj_cols] = df[obj_cols].astype("category")
        df[obj_cols] = df[obj_cols].apply(lambda x: x.cat.codes)
        return df

    def generate(self, write=True, clear=False, direct=False):
        self.mapping = Mapping(self.config, direct)
        if write:
            if direct:
                # output scipy objects
                writer = DirectWriter(self.mapping, self.config.output_folder)
            else:
                writer = Writer(self.mapping, self.config.output_folder, clear)
            writer.write()
