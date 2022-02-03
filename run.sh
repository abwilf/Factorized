#!/bin/bash

# BEST GRAPHQA
# python main.py \
# --drop_het 0.0 \
# --global_lr 0.001 \
# --bs 15 \
# --epochs 25 \
# --gat_conv_num_heads 4 \
# --graph_conv_in_dim 80 \
# --num_gat_layers 2 \
# --scene_mean 1 \
# --net graphqa \
# --out_dir /work/awilf/MTAG/results/graphqa \
# --trials 3



# # BEST FACTORIZED
python main.py \
--bs 10 \
--drop_het 0 \
--epochs 30 \
--gat_conv_num_heads 2 \
--global_lr 0.001 \
--graph_conv_in_dim 80 \
--net factorized \
--num_agg_nodes 1 \
--num_gat_layers 2 \
--scene_mean 1 \
--social_baseline 0 \
--test 0 \
--out_dir /work/awilf/MTAG/results/factorized \
--trials 3
