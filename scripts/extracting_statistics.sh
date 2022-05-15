#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

# Path to Datasets: [type(str)]
path_datasets="/srv/storage/datasets/eduardo/datasets"
# List of Datasets to Test: [PIPA, PISC]
datasets="PIPA PISC"
# List of Data Types: [domain, relationship]
types="domain relationship"
# Show Images: [--show]
show=""
# Debug mode: [--debug]
debug=""

for dataset in $datasets; do
    for type in $types; do
        python3 ../social_relation_recognition/extract_statistics.py \
        $path_datasets \
        $dataset \
        $type \
        $show \
        $debug  
    done
done