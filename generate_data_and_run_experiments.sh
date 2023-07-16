#!/bin/bash

set -e

# Generate data
docker build . -t ilargi:latest || echo "Building container failed"

echo "Generating data"
docker run \
    -v $(pwd)/resources/synthetic:/data/generated \
    --env CONFIG_WRITE_FOLDER=/data/generated/ \
    ilargi:latest python data_generator/generate.py  || echo "Data generation failed"


echo "Running experiments"


for cores in 8 16 24 32
do 
    echo "Running experiments with $cores cores"
    docker run \
        --cpus="$cores.0" \
        -v $(pwd)/resources/synthetic:/data/generated \
        -v $(pwd)/results/synthetic:/app/src/results \
        --env NUM_CORES=$cores \
        --env NUM_REPEATS=10 \
        --env DATA_GLOB_PATH=/data/generated/*/*/data/ \
        --env EXPERIMENT_TYPE=SYNTHETIC \
        ilargi:latest python experiments/experiment.py
done

echo "Succesfully completed synthetic data experiments"