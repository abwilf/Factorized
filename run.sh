#!/bin/bash

python main.py \
--bs 10 \
--drop_het 0 \
--epochs 30 \
--gat_conv_num_heads 4 \
--global_lr 0.001 \
--graph_conv_in_dim 60 \
--num_gat_layers 2 \
--scene_mean 1 \
--social_baseline 0 \
--solograph 1 \
--trials 3

