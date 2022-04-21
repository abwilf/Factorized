# Implementations for four types of graph leve augmentation: https://arxiv.org/pdf/2010.13902.pdf
# Node dropping
# Edge permutation
# Attribute masking (node)
# Subgraph sampling
# Please refer to the paper for more details. Code adapted from
# https://github.com/Shen-Lab/GraphCL/blob/master/unsupervised_TU/aug.py
import numpy as np
import torch

def drop_nodes(x, edge_index):
    raise NotImplementedError
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    return x, edge_index

def permute_edges(x, edge_index):
    device = x.device
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    permute_num = int(edge_num / 10)
    
    edge_index = edge_index.transpose(0, 1).detach().cpu().numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    edge_index = torch.tensor(edge_index).transpose_(0, 1).to(device)

    return x, edge_index


def subgraph(x, edge_index):
    raise NotImplementedError

    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    return x, edge_index


def mask_nodes(x):

    node_num, feat_dim = x.size()
    device = x.device
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32).to(device)

    return x

