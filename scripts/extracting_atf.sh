#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

# Path to Caffe Libraries: [type(str)]
path_caffe="/home/eduardo/Code/Caffe"
# Path to Pre-trained Models: [type(str)]
path_models="/srv/storage/datasets/eduardo/models"
# Path to Datasets: [type(str)]
path_datasets="/srv/storage/datasets/eduardo/datasets"
# List of Datasets to Test: [PIPA, PISC]
datasets="PIPA PISC"
# List of Data Types: [domain, relationship]
types="domain relationship"
# List of Features: [body_activity, body_age, body_clothing, body_gender, context_activity, context_emotion, object_attention]
features="body_activity body_age body_clothing body_gender context_activity context_emotion object_attention"
# Show Input Images: [--show]
show=""
# Debug Mode: [--debug]
debug=""

for dataset in $datasets; do
    for type in $types; do
        for feature in $features; do
            python2 ../social_relation_recognition/extract_attribute_features.py \
            $path_caffe \
            $path_models \
            $path_datasets \
            $dataset \
            $type \
            $feature \
            $show \
            $debug
        done
    done
done