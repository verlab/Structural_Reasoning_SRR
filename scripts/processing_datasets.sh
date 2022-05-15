#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

# Path to Datasets: [type(str)]
path_datasets="/srv/storage/datasets/eduardo/datasets"
# List of Datasets to Test: [PIPA, PISC]
datasets="PIPA PISC"
# Order JSON File: [--ordered]
ordered=""

for dataset in $datasets; do
    python3 ../social_relation_recognition/process_datasets.py \
    $path_datasets \
    $dataset \
    $ordered
done