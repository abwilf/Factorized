# hp = {
#     'epochs': [5],
#     'social_baseline': [1],
#     'trials': [3],
# }


## hp search over solograph model
# hp = {
#     'bs': [15],
#     'drop_het': [0,0.1,0.3],
#     'epochs': [25],
#     'gat_conv_num_heads': [2,4,6],
#     'global_lr': [0.001,.0001],
#     'graph_conv_in_dim': [60,80],
#     'num_gat_layers': [2,4,6],
#     'scene_mean': [1],
#     'solograph': [1],
#     'social_baseline': [0],
#     'trials': [3],
# }


hp = { # best performing solograph QA
    'drop_het': [0.0],
    'global_lr': [0.001],
    'bs': [15],
    'epochs': [25],
    'gat_conv_num_heads': [4],
    'graph_conv_in_dim': [80],
    'num_gat_layers': [2],
    'scene_mean': [1],
    'social_baseline': [0],
    'solograph': [1],
    'trials': [10],
}


# best performing factorized
# hp = {
#     'bs': [10],
#     'gat_conv_num_heads': [2],
#     'num_agg_nodes': [1],
#     'num_gat_layers': [2],
# }

# --bs 10 \
# --drop_het 0 \
# --epochs 30 \
# --gat_conv_num_heads 2 \
# --global_lr 0.001 \
# --graph_conv_in_dim 80 \
# --num_agg_nodes 1 \
# --num_gat_layers 2 \
# --scene_mean 1 \
# --social_baseline 0 \
# --solograph 1 \
# --trials 3

