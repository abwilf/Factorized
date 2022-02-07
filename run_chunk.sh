python main.py \
--bs 3 \
--drop_het 0 \
--epochs 15 \
--gat_conv_num_heads 2 \
--global_lr 0.001 \
--graph_conv_in_dim 80 \
--net factorized \
--num_agg_nodes 1 \
--num_gat_layers 2 \
--scene_mean 1 \
--social_baseline 0 \
--out_dir /work/awilf/MTAG/results/factorized \
--trials 1 \
--seq_len 25 \
--dataset social_unaligned \
--gran chunk \
--test 0 \
--zero_out_video 0 \
--zero_out_audio 0 \
--zero_out_text 0