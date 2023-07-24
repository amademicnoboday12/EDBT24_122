import configparser
import glob
import os
from pathlib import Path
from typing import List

from pathos import multiprocessing
import itertools
import numpy as np

from generator.generator import Generator
from tqdm import tqdm

from sigmod_datasets import DATASETS


def new_config(data_path):
    config = configparser.ConfigParser()
    # Allow the use of uppercase letters in config files
    config.optionxform = str
    config.add_section("Paths")
    config.set(
        "Paths",
        "output_folder",
        data_path,
    )
    config.add_section("Properties")
    return config

# SIGMOD CONFIG:
n_Rs = [1]
r_Ts = [100_000, 500_000, 1_000_000]  # data size
c_Ts = [10, 20, 30, 40, 50]
rho_c_Ss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
rho_c_Rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
ps = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t_Rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
rho_r_Ss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
ratios = [0.1, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.,
          5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.]

r_Xs = [10]


def write_config(base_path, n_R, r_T, c_T, rho_c_S, rho_c_R, p, jointype, t_R=None, rho_r_S=None):
    if rho_c_R > rho_c_S:
        # print(f'Config with path {base_path} skipped as rho_c_R is bigger than rho_c_S ({rho_c_R}>{rho_c_S})')
        return

    c_R = round((1 - rho_c_S + rho_c_R) * c_T, 1)
    temp = [None] * n_R
    modulo = c_R % n_R
    if modulo != 0:
        n_cols = ((c_R - modulo) / n_R)
        temp[0] = (n_cols + modulo) / c_T
        temp[1:] = [n_cols / c_T] * (n_R - 1)
    else:
        temp = [(c_R / n_R) / c_T] * n_R

    complexity_standard = r_T * c_T

    if rho_r_S != None:
        r_S = rho_r_S * r_T
    else:
        r_S = r_T
    c_S = c_T * rho_c_S
    c_R = c_T * (1 - rho_c_S + rho_c_R)

    common_properties = dict(
        join=jointype,
        r_T=str(int(r_T)),
        c_T=str(int(c_T)),
        p=str(p),
        rho_c_S=str(rho_c_S),
        rho_c_R=",".join([str(rho) for rho in temp]),
    )

    for ratio in ratios:
        config = configparser.ConfigParser()
        config.optionxform = str
        folder = f"{base_path}/ratio_{ratio}/"

        # For sigmod24 only keep preset configs
        if '-'.join(folder.split('-')[3:]) not in DATASETS:
            continue

        data_folder = f"{folder}data/"
        config.read_dict({
            "Properties": common_properties,
            "Paths": {"output_folder": data_folder}
        })

        path = f"{folder}/config.ini"

        complexity_factorized = complexity_standard / ratio

        r_R = (complexity_factorized - (r_S * c_S)) / c_R
        rho_r_R = r_R / r_T

        if rho_r_R <= 0 or rho_r_R > 1:
            print(f'Config with path {path} skipped as rho_r_R is ({rho_r_R}<=0 or {rho_r_R}> 1)')
            continue

        if r_R + r_S < r_T:
            print(f'Config with path {path} skipped as r_R + r_S < r_T is ({r_R} + {r_S} < {r_T}> 1)')

        config.set("Properties", "rho_r_R", ",".join([str(rho) for rho in [rho_r_R] * n_R]))

        if t_R is not None:
            t_T = ((r_S - r_R) * t_R) / r_T
            config.set("Properties", "t_T", str(t_T))

            if t_T < 0 or t_T > 1 or t_T > r_S:
                print(f'Config with path {path} skipped as not 0 < {t_T=} < 1 or t_T > {r_S=}')
                continue

        if rho_r_S is not None:
            config.set("Properties", "rho_r_S", str(rho_r_S))
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        # Write config file
        with open(path, "w+") as configfile:
            config.write(configfile)


def generate_data(configfile):
    try:
        generator = Generator(configfile)
        generator.generate(clear=True, direct=True)
    except Exception as E:
        # print(f"generating failed for {configfile}")
        ...


def generate_data_forall(glob_pattern):
    # for each config (.ini) file given the pattern create the dataset.
    files = list(glob.glob(glob_pattern))
    # print(files)
    processes = min(31, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=processes) as apool:
        for _ in tqdm(apool.imap(generate_data, files), total=len(files)):
            ...


def create_configs_sigmod(base_path: str):
    base_conf = dict(
        n_R=1,
        r_T=1000,
        c_T=10,
        p=0.0
    )
    configurations = []

    def create_conf(jointype, rho_c_S, rho_c_R, t_R=None, rho_r_S=None):
        params = dict(
            rho_c_S=rho_c_S,
            rho_c_R=rho_c_R,
            t_R=t_R,
            rho_r_S=rho_r_S,
            jointype=jointype
        )
        return {**base_conf, **params}

    # t_T > t_R?
    # no redundancy
    configurations.append(create_conf('inner', 0.3, 0.7))
    configurations.append(create_conf('left', 0.3, 0.7, 0.0))
    configurations.append(create_conf('outer', 0.3, 0.7, 0.0, 0.8))

    # source redundancy
    configurations.append(create_conf('inner', 0.5, 1.0))
    configurations.append(create_conf('left', 0.7, 0.7, 0.0))
    configurations.append(create_conf('outer', 0.7, 0.7, 0.0, 0.8))

    # target redundancy
    # configurations.append(create_conf('inner', 0.3, 0.7)) # Controlled by complexity
    configurations.append(create_conf('left', 0.7, 0.7, 0.7))
    configurations.append(create_conf('outer', 0.3, 0.7, 1.0, 0.4))

    # all redundancy
    configurations.append(create_conf('inner', 0.7, 0.7))
    # configurations.append(create_conf('left', 0.7, 0.7, 0.7))
    configurations.append(create_conf('outer', 0.7, 0.7, 1.0, 0.4))

    for config in tqdm(configurations):
        conf_string = '-'.join(
            [
                f"{key}={value}" for key, value in
                config.items()
            ]
        )
        # print(conf_string)
        write_config(base_path + conf_string, **config)


def create_configs(base_path: str, jointypes: List, ):
    """
    Create all configs (product) for the given global data characteristics values
    """
    if jointypes is None:
        jointypes = ['inner', 'left', 'outer']

    configurations = list()
    if 'inner' in jointypes:
        configurations += list(itertools.product(n_Rs, r_Ts, c_Ts, rho_c_Ss, rho_c_Rs, ps, ['inner']))
    if 'left' in jointypes:
        configurations += list(itertools.product(n_Rs, r_Ts, c_Ts, rho_c_Ss, rho_c_Rs, ps, ['left'], t_Rs))
    if 'outer' in jointypes:
        configurations += list(itertools.product(n_Rs, r_Ts, c_Ts, rho_c_Ss, rho_c_Rs, ps, ['outer'], t_Rs, rho_r_Ss))

    print(f"Creating {len(configurations)} configurations")
    for config in tqdm(configurations):
        conf_string = '-'.join(
            [
                f"{key}={value}" for key, value in
                zip(
                    ["n_R", "r_T", "c_T", "rho_c_S", "rho_c_R", "p", "join", "t_R", "rho_r_S"][:len(config)],
                    config
                )
            ]
        )
        write_config(base_path + conf_string, *config)


if __name__ == "__main__":
    configpath = os.environ.get("CONFIG_WRITE_FOLDER") or "./configs/"
    create_configs(configpath, ['inner', 'left', 'outer'])
    generate_data_forall(f"{configpath}/*/*/*.ini")
