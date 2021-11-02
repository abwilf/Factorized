import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from datetime import datetime
import json
import os
import sys
import time
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import gc as g
from graph_model.iemocap_inverse_sample_count_ce_loss import IEMOCAPInverseSampleCountCELoss
from model import NetMTGATAverageUnalignedConcatMHA
from dataset.MOSEI_dataset import MoseiDataset
from dataset.MOSEI_dataset_unaligned import MoseiDatasetUnaligned
from dataset.MOSI_dataset import MosiDataset
from dataset.MOSI_dataset_unaligned import MosiDatasetUnaligned
from dataset.IEMOCAP_dataset import IemocapDatasetUnaligned, IemocapDataset
import logging
import util
import pathlib
import random
from arg_defaults import defaults
from consts import GlobalConsts as gc

from alex_utils import *
import standard_grid

import gc as g
from sklearn.metrics import accuracy_score

from torch_geometric.nn import Linear
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from torch_scatter import scatter_mean
import torch.nn.functional as F
from gatv3conv import GATv3Conv
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from graph_builder import construct_time_aware_dynamic_graph, build_time_aware_dynamic_graph_uni_modal, build_time_aware_dynamic_graph_cross_modal

def set_seed(my_seed):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)

set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_map = {
    'mosi': MosiDataset,
    'mosi_unaligned': MosiDatasetUnaligned,
    'mosei': MoseiDataset,
    'mosei_unaligned': MoseiDatasetUnaligned,
    'iemocap_unaligned': IemocapDatasetUnaligned,
    'iemocap': IemocapDataset
}

ie_emos = ["Neutral", "Happy", "Sad", "Angry"]

# get all connection types for declaring heteroconv later
mods = ['text', 'audio', 'video']
conn_types = ['past', 'pres', 'fut']
all_connections = []
for mod in mods:
    for mod2 in mods:
        for conn_type in conn_types:
            all_connections.append((mod, conn_type, mod2))


def topk_edge_pooling(percentage, edge_index, edge_weights):
    if percentage < 1.0:
        p_edge_weights = torch.mean(edge_weights, 1).squeeze()
        sorted_inds = torch.argsort(p_edge_weights, descending=True)
        kept_index = sorted_inds[:int(len(sorted_inds) * percentage)]
        # kept = p_edge_weights >= self.min_score
        return edge_index[:, kept_index], edge_weights[kept_index], kept_index
    else:
        return edge_index, edge_weights, torch.arange(edge_index.shape[1]).to(edge_index.device)

def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def weighted_acc(preds, truths):
    preds, truths = preds > 0, truths > 0
    tn, fp, fn, tp = confusion_matrix(truths, preds).ravel()
    n, p = len([i for i in preds if i == 0]), len([i for i in preds if i > 0])
    return (tp * n / p + tn) / (2 * n)

def eval_iemocap(split, output_all, label_all, epoch=None):
    truths = np.array(label_all)
    results = np.array(output_all)
    test_preds = results.reshape((-1, 4, 2))
    test_truth = truths.reshape((-1, 4))
    emos_f1 = {}
    emos_acc = {}
    for emo_ind, em in enumerate(ie_emos):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        emos_f1[em] = f1
        acc = accuracy_score(test_truth_i, test_preds_i)
        emos_acc[em] = acc
    
    return {
        'f1': emos_f1,
        'acc': emos_acc
    }

def eval_mosi_mosei(split, output_all, label_all):
    truth = np.array(label_all)
    preds = np.array(output_all)
    mae = np.mean(np.abs(truth - preds))
    cor = np.corrcoef(preds, truth)[0][1]
    acc = accuracy_score(truth >= 0, preds >= 0)
    non_zeros = np.array([i for i, e in enumerate(truth) if e != 0])
    ex_zero_acc = accuracy_score((truth[non_zeros] > 0), (preds[non_zeros] > 0))

    preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    truth_a7 = np.clip(truth, a_min=-3., a_max=3.)
    acc_7 = multiclass_acc(preds_a7, truth_a7)

    # F1 scores. All of them are recommended by previous work.
    f1_mfn = f1_score(np.round(truth), np.round(preds), average="weighted")  # We don't use it, do we?
    f1_raven = f1_score(truth >= 0, preds >= 0, average="weighted")  # Non-negative VS. Negative
    f1_mult = f1_score((truth[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')  # Positive VS. Negative

    return {
        'mae': mae,
        'corr': cor,
        'acc_2': acc,
        'acc_7': acc_7,
        'ex_zero_acc': ex_zero_acc,
        'f1_raven': f1_raven, # includes zeros, Non-negative VS. Negative
        'f1_mult': f1_mult,  # exclude zeros, Positive VS. Negative
    }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # Added to support odd d_model
        pe = pe.unsqueeze(0).transpose(0, 1).squeeze()
        self.register_buffer('pe', pe)

    def forward(self, x, counts):
        pe_rel = torch.cat([self.pe[:count,:] for count in counts])
        x = x + pe_rel.to(device)
        return self.dropout(x)

def get_masked(arr):
    if (arr==0).all():
        return torch.tensor([]).long()
    else:
        return arr[torch.argmax(~((arr==0).all(dim=-1)).to(torch.long)):]

def get_loader(ds):
    words = ds[:][0]
    covarep = ds[:][1]
    facet = ds[:][2]

    total_data = []
    for i in range(words.shape[0]):
        data = {
            'text': get_masked(words[i]),
            'audio': get_masked(covarep[i]),
            'video': get_masked(facet[i]),
        }
        
        if sum([len(v) for v in data.values()]) == 0:
            continue
        
        hetero_data = { k: {'x': v} for k,v in data.items()}
        
        data = {
            **data,
            'text_idx': torch.arange(data['text'].shape[0]),
            'audio_idx': torch.arange(data['audio'].shape[0]),
            'video_idx': torch.arange(data['video'].shape[0]),
        }
        
        for mod in mods:
            ret = build_time_aware_dynamic_graph_uni_modal(data[f'{mod}_idx'],[], [], 0, all_to_all=gc['use_all_to_all'], time_aware=True, type_aware=True)
            
            if len(ret) == 0: # no data for this modality
                continue
            elif len(ret) == 1:
                data[mod, 'pres', mod] = ret[0]
            else:
                data[mod, 'pres', mod], data[mod, 'fut', mod], data[mod, 'past', mod] = ret

            for mod2 in [modx for modx in mods if modx != mod]: # other modalities
                ret = build_time_aware_dynamic_graph_cross_modal(data[f'{mod}_idx'],data[f'{mod2}_idx'], [], [], 0, time_aware=True, type_aware=True)
                
                if len(ret) == 0:
                    continue
                if len(ret) == 2: # one modality only has one element
                    if len(data[f'{mod}_idx']) > len(data[f'{mod2}_idx']):
                        data[mod2, 'pres', mod], data[mod, 'pres', mod2] = ret
                    else:
                        data[mod, 'pres', mod2], data[mod2, 'pres', mod] = ret
                
                else:
                    if len(data[f'{mod}_idx']) > len(data[f'{mod2}_idx']): # the output we care about is the "longer" sequence
                        ret = ret[3:]
                    else:
                        ret = ret[:3]

                    data[mod, 'pres', mod2], data[mod, 'fut', mod2], data[mod, 'past', mod2] = ret
                 
        # quick assertions
        for mod in mods:
            assert isinstance(data[mod], torch.Tensor)
            for mod2 in [modx for modx in mods if modx != mod]:
                if (mod, 'fut', mod2) in data:
                    assert (data[mod, 'fut', mod2].flip(dims=[0]) == data[mod2, 'past', mod]).all()
                    assert isinstance(data[mod, 'fut', mod2], torch.Tensor) and isinstance(data[mod, 'past', mod2], torch.Tensor) and isinstance(data[mod, 'pres', mod2], torch.Tensor)

        hetero_data = {
            **hetero_data,
            **{k: {'edge_index': v} for k,v in data.items() if isinstance(k, tuple) }
        }
        hetero_data = HeteroData(hetero_data)
        
        # hetero_data = T.AddSelfLoops()(hetero_data) # todo: include this as a HP to see if it does anything!
        hetero_data.y = ds[i][-1]
        total_data.append(hetero_data)

    loader = DataLoader(total_data, batch_size=gc['batch_size'], shuffle=True)
    # loader = DataLoader(total_data, batch_size=gc['batch_size'], shuffle=False)
    # loader = DataLoader(total_data, batch_size=2)
    return loader


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = 4
        
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in mods:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            # for mod in mods:
            #     self.lin_layers[f'{i}_{mod}'] = Linear(64, (hidden_channels//self.heads)*self.heads, bias=True, weight_initializer='glorot')
            
            # conv = HeteroConv({
            #     conn_type: GATv2Conv(64, hidden_channels//self.heads, heads=self.heads)
            #     for conn_type in all_connections
            # }, aggr='mean')
            

            # UNCOMMENT FOR PARAMETER SHARING
            mods_seen = {} # mapping from mod to the gatv3conv linear layer for it
            d = {}
            for conn_type in all_connections:
                mod_l, _, mod_r = conn_type

                lin_l = None if mod_l not in mods_seen else mods_seen[mod_l]
                lin_r = None if mod_r not in mods_seen else mods_seen[mod_r]

                _conv =  GATv3Conv(
                    lin_l,
                    lin_r,
                    64, 
                    hidden_channels//self.heads,
                    heads=self.heads
                )
                if mod_l not in mods_seen:
                    mods_seen[mod_l] = _conv.lin_l
                if mod_r not in mods_seen:
                    mods_seen[mod_r] = _conv.lin_r
                d[conn_type] = _conv
            
            conv = HeteroConv(d, aggr='mean')

            self.convs.append(conv)

        self.finalW = nn.Sequential(
            Linear(-1, hidden_channels // 4),
            nn.ReLU(),
            # nn.Linear(hidden_channels // 4, label_dim),
            Linear(hidden_channels // 4, hidden_channels // 4),
            nn.ReLU(),
            Linear(hidden_channels // 4, out_channels),
        )
        
        self.pes = {k: PositionalEncoding(64) for k in mods}

    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}

        # apply pe
        for m, v in x_dict.items(): # modality, tensor
            idxs = batch_dict[m]
            assert (idxs==(idxs.sort().values)).all()
            _, counts = torch.unique(idxs, return_counts=True)
            x_dict[m] = self.pes[m](v, counts)

        for conv in self.convs:
            # x_dict = conv(x_dict, edge_index_dict)
            x_dict, edge_types = conv(x_dict, edge_index_dict, return_attention_weights_dict={elt: True for elt in all_connections})

            '''
            x_dict: {
                modality: (
                    a -> tensor of shape batch_num_nodes (number of distinct modality nodes concatenated from across whole batch),
                    b -> [
                    (
                        edge_idxs; shape (2, num_edges) where num_edges changes depending on edge_type (and pruning),
                        attention weights; shape (num_edges, num_heads)
                    )
                    ] of length 9 b/c one for each edge type where text modality is dst, in same order as edge_types[modality] list
                )
            }
            '''

            attn_dict = {
                k: {
                    edge_type: {
                        'edge_index': edge_index,
                        'edge_weight': edge_weight,
                    }
                    for edge_type, (edge_index, edge_weight) in zip(edge_types[k], v[1])
                } 
                for k, v in x_dict.items()
            }

            x_dict = {key: x[0].relu() for key, x in x_dict.items()}

        # readout: avg nodes (no pruning yet!)
        x = torch.cat([v for v in x_dict.values()], axis=0)
        batch_dicts = torch.cat([v for v in batch_dict.values()], axis=0)
        x = scatter_mean(x,batch_dicts, dim=0)
        return self.finalW(x).squeeze(axis=-1)

def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(train_loader, model, optimizer):
    total_loss, total_examples = 0,0
    y_trues = []
    y_preds = []
    model.train()
    for batch_i, data in enumerate(tqdm(train_loader)): # need index to prune edges
        if 'iemocap' in gc['dataset']:
            data.y = data.y.reshape(-1,4)

        if data.num_edges > 1e6:
            print('Data too big to fit in batch')
            continue
            
        cont = False
        for mod in mods:
            if not np.any([mod in elt for elt in data.edge_index_dict.keys()]):
                print(mod, 'dropped from train loader!')
                cont = True
        if cont:
            continue
        
        data = data.to(device)
        if batch_i == 0:
            with torch.no_grad():  # Initialize lazy modules.
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict)

        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        if 'iemocap' in gc['dataset']:
            loss = nn.CrossEntropyLoss()(out, data.y.argmax(-1))
        else:
            loss = F.mse_loss(out, data.y)
            loss = loss / torch.abs(loss.detach()) # norm

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        total_examples += data.num_graphs

        y_true = data.y.detach().cpu().numpy()
        y_pred = out.detach().cpu().numpy()
        
        y_trues.extend(y_true)
        y_preds.extend(y_pred)

        del loss
        del out
        del data

    torch.cuda.empty_cache()
    return total_loss / total_examples, y_trues, y_preds

@torch.no_grad()
def test(loader, model, scheduler, valid):
    y_trues = []
    y_preds = []
    model.eval()

    l = 0.0
    for batch_i, data in enumerate(loader):
        cont = False
        for mod in mods:
            if not np.any([mod in elt for elt in data.edge_index_dict.keys()]):
                print(mod, 'dropped from test loader!')
                cont = True
        if cont:
            continue

        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        if 'iemocap' in gc['dataset']:
            loss = nn.CrossEntropyLoss()(out, data.y.argmax(-1))
        else:
            loss = F.mse_loss(out, data.y)
            loss = loss / torch.abs(loss.detach()) # norm
        l += F.mse_loss(out, data.y, reduction='mean').item()

        y_true = data.y.detach().cpu().numpy()
        y_pred = out.detach().cpu().numpy()
        
        y_trues.extend(y_true)
        y_preds.extend(y_pred)

        del data
        del out
    
    # if valid:
    #     scheduler.step(mse)
    return mse, y_trues, y_preds
    

def train_model(optimizer, use_gnn=True, exclude_vision=False, exclude_audio=False, exclude_text=False, average_mha=False, num_gat_layers=1, lr_scheduler=None, reduce_on_plateau_lr_scheduler_patience=None, reduce_on_plateau_lr_scheduler_threshold=None, multi_step_lr_scheduler_milestones=None, exponential_lr_scheduler_gamma=None, use_pe=False, use_prune=False):
    assert lr_scheduler in ['reduce_on_plateau', 'exponential', 'multi_step',
                            None], 'LR scheduler can only be [reduce_on_plateau, exponential, multi_step]!'

    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ds = dataset_map[gc['dataset']]
    train_dataset = ds(gc['data_path'], clas="train")
    test_dataset = ds(gc['data_path'], clas="test")
    valid_dataset = ds(gc['data_path'], clas="valid")

    train_loader, train_labels = get_loader(train_dataset), train_dataset[:][-1]
    valid_loader, valid_labels = get_loader(valid_dataset), valid_dataset[:][-1]
    test_loader, test_labels = get_loader(test_dataset), test_dataset[:][-1]

    out_channels = 4 if 'iemocap' in gc['dataset'] else 1
    model = HeteroGNN(hidden_channels=64, out_channels=out_channels, num_layers=6)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=gc['global_lr'],
        weight_decay=gc['weight_decay'],
        betas=(gc['beta1'], gc['beta2']),
        eps=gc['eps']
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=.002)

    eval_fns = {
        'mosi_unaligned': eval_mosi_mosei,
        'mosei_unaligned': eval_mosi_mosei,
        'iemocap_unaligned': eval_iemocap,
    }
    eval_fn = eval_fns[gc['dataset']]

    best_mae = 1
    best_test_acc = 0
    best_valid_acc = 0
    for epoch in range(gc['epoch_num']):
        loss, y_trues_train, y_preds_train = train(train_loader, model, optimizer)
        train_res = eval_fn('train', y_preds_train, y_trues_train)
        train_acc, train_mae = train_res['acc_2'], train_res['mae']

        valid_loss, y_trues_valid, y_preds_valid = test(valid_loader, model, scheduler, valid=True)
        valid_res = eval_fn('valid', y_preds_valid, y_trues_valid)
        valid_acc, valid_mae = valid_res['acc_2'], valid_res['mae']

        test_loss, y_trues_test, y_preds_test = test(test_loader, model, scheduler, valid=False)
        test_res = eval_fn('test', y_preds_test, y_trues_test)
        test_acc, test_mae = test_res['acc_2'], test_res['mae']

        if valid_acc < best_valid_acc:
            best_mae = mae
            best_test_acc = test_acc
            best_valid_acc = valid_acc

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, ' f'Valid: {valid_acc:.4f}, Test: {test_acc:.4f}')

    print(f'\nBest test acc: {best_test_acc:.4f},\nBest valid acc: {best_valid_acc:.4f}')
    print('Model parameters:', count_params(model))


def get_arguments():
    parser = standard_grid.ArgParser()
    for arg in defaults:
        parser.register_parameter(*arg)

    args = parser.compile_argparse()

    global gc
    for arg, val in args.__dict__.items():
        gc[arg] = val

if __name__ == "__main__":
    get_arguments() # updates gc

    assert gc['dataroot'] is not None, "You havn't provided the dataset path! Use the default one."
    assert gc['task'] in ['mosi', 'mosei', 'mosi_unaligned', 'mosei_unaligned', 'iemocap', 'iemocap_unaligned'], "Unsupported task. Should be either mosi or mosei"

    gc['data_path'] = gc['dataroot']
    gc['dataset'] = gc['task']

    if not gc['eval']:
        start_time = time.time()
        util.set_seed(gc['seed'])
        best_results = train_model(gc['optimizer'],
                                   use_gnn=gc['useGNN'],
                                   average_mha=gc['average_mha'],
                                   num_gat_layers=gc['num_gat_layers'],
                                   lr_scheduler=gc['lr_scheduler'],
                                   reduce_on_plateau_lr_scheduler_patience=gc['reduce_on_plateau_lr_scheduler_patience'],
                                   reduce_on_plateau_lr_scheduler_threshold=gc['reduce_on_plateau_lr_scheduler_threshold'],
                                   multi_step_lr_scheduler_milestones=gc['multi_step_lr_scheduler_milestones'],
                                   exponential_lr_scheduler_gamma=gc['exponential_lr_scheduler_gamma'],
                                   use_pe=gc['use_pe'],
                                   use_prune=gc['use_prune'])
        elapsed_time = time.time() - start_time
        out_dir = "output/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        res_f = open(os.path.join(out_dir, "best.txt"), "w")
        res_f.write(json.dumps(best_results))
    
    else:
        assert gc['resume_pt'] is not None
        log_path = os.path.dirname(os.path.dirname(gc['resume_pt']))
        log_file = os.path.join(log_path, 'eval.log')
        logging.basicConfig(level=logging.INFO)
        logging.getLogger().addHandler(logging.FileHandler(log_file))
        # logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Start evaluation Using model from {}".format(gc['resume_pt']))
        start_time = time.time()
        eval_model(gc['resume_pt'])
        logging.info("Total evaluation time: {}".format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        )
