import glob
from tqdm import tqdm

from data_generator.generator.generator import Generator

"""
Generate test data from configs stored in tests/resources/generated/configs.
Meant to be run from `tests` directory:
    python resources/generated/generate_data.py
"""

config_path: str
for config_path in tqdm(glob.glob("./resources/generated/configs/*/*.ini")):
    Generator(config_path).generate(write=True, clear=True, direct=True)