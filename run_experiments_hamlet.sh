#!/bin/bash

set -e


docker build . -t ilargi:latest || echo "Building container failed"

echo "Running experiments with Hamlet data"


for cores in 8 16 24 32
do 
    echo "Running experiments with $cores cores"
    docker run \
        --cpus="$cores.0" \
        -v $(pwd)/results/hamlet:/app/src/results \
        --env NUM_CORES=$cores \
        --env NUM_REPEATS=10 \
        --env EXPERIMENT_TYPE=HAMLET \
        --env DATA_ROOT=/data/hamlet \
        ilargi:latest python experiments/experiment.py 
done

echo "Succesfully completed experiments with hamlet data"
