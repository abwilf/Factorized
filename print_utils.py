# ## script to hp
# a = '''

# python main.py \
# --bs 10 \
# --drop_het 0 \
# --epochs 25 \
# --gat_conv_num_heads 2 \
# --global_lr 0.001 \
# --graph_conv_in_dim 80 \
# --num_agg_nodes 1 \
# --num_gat_layers 2 \
# --scene_mean 1 \
# --social_baseline 0 \
# --solograph 1 \
# --trials 1 \
# --hi hello

# '''
# stripped = a.strip().split('main.py')[1].strip()
# keys = stripped.split()[0::2]
# vals = stripped.split()[1::2]
# d = {k.replace('--', ''):[(int(v) if v.isnumeric() else v)] for (k,v) in zip(keys,vals)}
# d = [elt.strip() for elt in str(d)[1:-1].split(',')]
# s = 'hp = {'
# for elt in d:
#     s += f'\n  {elt},'
# s += '\n}'
# print(s)



## hp to script
d = {
 'bs': [10],
 'drop_het': [0],
 'epochs': [25],
 'gat_conv_num_heads': [2],
 'global_lr': ['0.001'],
 'graph_conv_in_dim': [80],
 'num_agg_nodes': [1],
 'num_gat_layers': [2],
 'scene_mean': [1],
 'social_baseline': [0],
 'solograph': [1],
 'trials': [1],
 'hi': ['hello']
}

assert min([len(v)==1 for v in d.values()]) == 1 # each array of len 1
s = 'python main.py \\'
for k,v in d.items():
    s += f'\n--{k} {v[0]} \\'

s = s[:-1] + '\n' # get rid of last \
print(s)



