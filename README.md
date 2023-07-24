# Ilargi

## Repo layout
 - [`results`](./results) Experiment results
 - [`resources`](./resources) Data
 - [`src`](./src) Ilargi Source files
    - [`cost_estimation`](./src/cost_estimation/) Cost estimation source files
    - [`data_generator`](./src/data_generator/) Data generator
    - [`experiments`](./src/experiments/) Experiment runner and configs
    - [`factorization`](./src/factorization/) Ilargi data structure and model sources

## Requirements
 - Docker
 - 1TB of free space for running experiments with synthetic data
 - 32 CPU cores (if you want to run the full range of experiments)

## Recreating Ilargi results
**Only tested on linux** 
**Run all scripts from the repo root** 

To run a script in the background and print output to a file:
`nohup <./script.sh> > <logfile> &`
### Synthetic data results

!This script will generate approximately 1TB of data and write it to the `./resources/synthetic` directory!

!This will take very long as it will run all experiments sequentially!

Run `./generate_data_and_run_experiments.sh`. This will execute the folowing actions:
1. Build the docker container
1. Create the data config files in `./resources/synthetic/...`
1. Genreate the synthetic data.
1. Run all experiments (8, 16, 24, 32 cores) and write results to `./results/synthetic`


### Real data results: Hamlet
Run `run_hamlet_experiments.sh` which will run the experiments and write results to `./results/hamlet`



