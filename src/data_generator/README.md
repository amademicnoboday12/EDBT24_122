Adapted from the [Valentine Data Fabricator](https://github.com/delftdata/valentine-data-fabricator).
# How it works
The data generator is a tool that can create synthetic data integration scenarios, used for testing the performance of factorized machine learning. The generator can produce data scenarios with different levels of target and source redundancy, sparsity, and jointypes.

## Input parameters
The generator takes the following input parameters:
| Symbol | Parameter | Description |
| --- | --- | --- |
| $c_{T}$, $c_{r_k}$, $r_T$, $r_{r_k}$ | `c_T`, `r_T`, `r_R`, `c_R` | Specifies the size of the target table. |
|  | Inferred from length of `r_R` | Specifies how many source tables are involved in the data integration scenario. |
| $j$ | `join` |  Specifies the type of join operation (inner, left, or full outer) between the source tables. |
| $\rho_c(S)$ | `rho_c_S`  | Specifies how many columns the Entity table should have relative to the target table. |
| $\rho_r(R)$  | `rho_r_R`| Specifies how many rows the attribute tables should have relative to the target table. |
| $\rho_c(R)$ | `rho_c_R` | Specifies how many columns the attribute tables should have relative to the target table. |
| $t_T$| `t_T` | (not for inner join) Specifies fraction of rows in the target table should have redundant data. |
|$\rho_R(S)$| `rho_r_S`| (only outer join) Specifies how many rows the Entity table should have relative to the target table.|
| $p$ | `p` | Specifies how sparse the Target table should be. |


## The generation process
The generator works by creating a normalized matrix representation of the data integration scenario, consisting of an entity table S and a set of attribute tables R. The generator uses random values to fill the source tables, and creates indicator and mapping matrices to describe how the source tables are related to each other and to the target table. The generator also creates row and column mappings between the normalized and materialized data, which can be written to files or directly to SciPy sparse matrices. 

The more detailed process, which can be found in [mapping.py](generator/mapping.py) is:
1. Create list of columns matches for the source tables
2. Create list of row (index) matches for the source tables
3. Add non matching, columns and rows
4. Fill the tables with intermediate values
5. Create (intermediate) T by joining the source tables
6. Iterate over all tables and fill with random data, taking into account the row/col mappings and simultaneously keeping track of the row and column mappings.


## Output datasets
Generating a dataset will result in multiple scipy sparse matrices (`.mtx`) being written to the target directory, namely:
- `Sk.mtx` for each source/attribute table
- `T.mtx` the Target table
- `T_Sk_row_mapping.mtx` the row mappings for each source table
- `T_Sk_col_mapping.mtx` the column mappings for each source table

# Usage
##  1. <a name='Prerequisites'></a>Prerequisites

To run the generator you need:

- Python 3.8.x

##  2. <a name='ExecutionInstructions'></a>Execution Instructions

To run the generator the following steps should be followed:

- Run `pip3 install .`.
- Create a target and a source folder, named accordingly.
- Fill in your settings in the config file `config.ini`.
- Running from terminal:
    - Run the generator with the command: `python generator.py config.ini`. Depending on your python setup python3.8 or python3 might need to be used instead of python.
    - Run all configuration files available using `generate_scripts/generate_all.sh`.
- Running from code:
    - Change your directory to the `amalur-data-generator` folder.
    - Import the generator module using `from generator.generator import Generator`.
    - Create data using `Generator(config.ini).generate()`.




