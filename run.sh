#!/bin/bash

python main.py \
--bs 15 \
--drop_het 0 \
--epochs 50 \
--gat_conv_num_heads 4 \
--global_lr 0.001 \
--graph_conv_in_dim 64 \
--num_gat_layers 2 \
--scene_mean 1 \
--solograph 1 \
