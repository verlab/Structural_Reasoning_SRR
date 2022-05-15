#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

###########################
##          RUN          ##
###########################
# Metric: [acc, map]
metric='map'
###########################
##         PATHS         ##
###########################
# Path to Configurations: [type(str)]
path_configs='/srv/storage/datasets/eduardo/results/Pytorch/SGN/PISC/relationship/SKG/3x/fusion/test/logs/configs_test.yaml'
# Path to Features: [type(str)]
path_datasets='/srv/storage/datasets/eduardo/datasets/'
# Path to Save: [type(str)]
path_models='/srv/storage/datasets/eduardo/models/Pytorch/SGN/'
# Path to Results: [type(str)]
path_results='/srv/storage/datasets/eduardo/results/Pytorch/SGN/'
###########################
##         MODES         ##
###########################
# Cuda mode: [--cuda]
cuda='--cuda'
# Debug mode: [--debug]
debug=''
# Show mode: [--show]
show=''
# Test mode: [--test]
test=''

python3 ../social_relation_recognition/test_model.py \
    $metric \
    $path_configs \
    $path_datasets \
    $path_models \
    $path_results \
    $cuda \
    $debug \
    $show \
    $test