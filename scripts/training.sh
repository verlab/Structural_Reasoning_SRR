#!/bin/bash

cd "$( cd "$( dirname "$0" )" && pwd )"

###########################
##          RUN          ##
###########################
# Model Name: [type(str)]
name='test'
# Dataset Type: [PIPA, PISC]
dataset='PISC'
# Relation Type: [domain, relationship]
relation='relationship'
# Final Loss: [sum, weighted]
loss='sum'
# Dataset Statistics: [ImageNet, Self]
statistics='ImageNet'
# Freeze Parameters: [all, conv, finetune, norm, none]
freeze='finetune'
# Class Weights: [false, true]
weights='true'
# Adam AMSGrad: [false, true]
amsgrad='false'
# Random Seed: [type(int)]
seed=27
###########################
##         GRAPH         ##
###########################
# Graph Topology: [SKG-, SKG, SKG+]
topology='SKG'
# Social Scales: [personal, local, global]
scales='personal local global'
# Relation Edges: [neighbors, full, none]
edges='full'
###########################
##         MODEL         ##
###########################
# Attributes
# Convolution: [gcn, none]
attributes_conv='gcn'
# Activation: [elu, leaky_relu, relu, selu, sigmoid, tanh, none]
attributes_activation='tanh'
# Aggregation: [lse, max, mean, min, sum, none]
attributes_aggregation='lse'
# Normalization: [batch, layer, none]
attributes_normalization='none'
# Dropout Rate: [type(float)]
attributes_dropout=0.0
# Scales
# Convolution: [3x, 4x]
scales_conv='3x'
# Activation: [elu, leaky_relu, relu, selu, sigmoid, tanh, none]
scales_activation='relu'
# Aggregation: [lse, max, mean, min, sum, none]
scales_aggregation='sum'
# Normalization: [batch, layer, none]
scales_normalization='none'
# Dropout Rate: [type(float)]
scales_dropout=0.0
# Relations
# Convolution: [fusion, gat, ggcn]
relations_conv='fusion'
# Activation: [elu, leaky_relu, relu, selu, sigmoid, tanh, none]
relations_activation='relu'
# Aggregation: [lse, max, mean, min, sum, none]
relations_aggregation='none'
# Normalization: [batch, layer, none]
relations_normalization='layer'
# Dropout Rate: [type(float)]
relations_dropout=0.0
###########################
##    HYPERPARAMETERS    ##
###########################
# Workers Number: [type(int)]
workers=8
# Epochs Number: [type(int)]
epochs=15
# Input Size: [type(int)]
input=224
# Batch Size: [type(int)]
batch=8
# Hidden State Size: [type(int)]
hidden=512
# Learning Rate: [type(float)]
learning=0.0002
# Weight Regularization: [type(float)]
regularization=0.00002
# SGD Momentum: [type(float)]
momentum=0.9
# Decay Rate: [type(float)]
decay=0.98
###########################
##         PATHS         ##
###########################
# Path to Features: [type(str)]
path_datasets='/srv/storage/datasets/eduardo/datasets/'
# Path to Save: [type(str)]
path_models='/srv/storage/datasets/eduardo/models/Pytorch/SGN/'
# Path to Results: [type(str)]
path_results='/srv/storage/datasets/eduardo/results/Pytorch/SGN/'
###########################
##         MODES         ##
###########################
# Debug mode: [--debug]
debug=''
# Eval mode: [--eval]
eval='--eval'
# Save mode: [--save]
save=''

python3 ../social_relation_recognition/train_model.py \
    $name \
    $dataset \
    $relation \
    $loss \
    $statistics \
    $freeze \
    $weights \
    $amsgrad \
    $seed \
    $topology \
    $scales \
    $edges \
    $attributes_conv \
    $attributes_activation \
    $attributes_aggregation \
    $attributes_normalization \
    $attributes_dropout \
    $scales_conv \
    $scales_activation \
    $scales_aggregation \
    $scales_normalization \
    $scales_dropout \
    $relations_conv \
    $relations_activation \
    $relations_aggregation \
    $relations_normalization \
    $relations_dropout \
    $workers \
    $epochs \
    $input \
    $batch \
    $hidden \
    $learning \
    $regularization \
    $momentum \
    $decay \
    $path_datasets \
    $path_models \
    $path_results \
    $debug \
    $eval \
    $save