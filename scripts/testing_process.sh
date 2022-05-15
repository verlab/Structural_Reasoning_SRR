#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

# Path to Datasets: [type(str)]
path_datasets="/srv/storage/datasets/eduardo/datasets"
# List of Datasets to Test: [PIPA, PISC]
datasets="PIPA PISC"

for dataset in $datasets; do
    if [ $dataset == "PIPA" ]; then
        python3 ../social_relation_recognition/test_metadata.py \
            $path_datasets \
            $dataset 
    else
        python2 ../social_relation_recognition/test_metadata.py \
            $path_datasets \
            $dataset
    fi
done