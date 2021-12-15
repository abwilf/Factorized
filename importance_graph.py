from collections import OrderedDict
from torch.autograd import Variable

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

SG_PATH = '/work/awilf/Standard-Grid'
import sys
sys.path.append(SG_PATH)
import standard_grid

import gc as g
from sklearn.metrics import accuracy_score

from torch_geometric.nn import Linear

from hetero_conv import HeteroConv
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from torch_scatter import scatter_mean
import torch.nn.functional as F
from gatv3conv import GATv3Conv
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from HeteroDataLoader import DataLoader
import torch_geometric.transforms as T

from graph_builder import construct_time_aware_dynamic_graph, build_time_aware_dynamic_graph_uni_modal, build_time_aware_dynamic_graph_cross_modal

import mylstm
from social_iq import *

SEEDS = list(range(100))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gc = {}

dataset_map = {
    'mosi': MosiDataset,
    'mosi_unaligned': MosiDatasetUnaligned,
    'mosei': MoseiDataset,
    'mosei_unaligned': MoseiDatasetUnaligned,
    'iemocap_unaligned': IemocapDatasetUnaligned,
    'iemocap': IemocapDataset,
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
mod_conns = [elt for elt in all_connections if not ( (elt[0]=='q' and elt[-1]=='a') or (elt[0]=='a' and elt[-1]=='q') ) ]

def get_fc_combinations(idxs_a, idxs_b): # get array of shape (2, len(idxs_a)*len(idxs_b)) for use in edge_index
    if len(idxs_a) == 0 or len(idxs_b) == 0:
        return torch.zeros((2,0))
    
    return torch.from_numpy(np.array(np.meshgrid(idxs_a, idxs_b)).reshape((-1, len(idxs_a)*len(idxs_b)))).to(torch.long)


@memoized
def get_idxs(a, b, conn_type):
    '''
    a is the length of the indices array for src, b is same for tar
    get all indeces between a (src) and b (tar) according to conn_type.  if present, only choose indices that match.  if past, all a indices must be > b indices
    '''
    a = np.arange(a)
    b = np.arange(b)
    
    tot = np.array(list(product(a,b)))
    a_idxs, b_idxs = tot[:,0], tot[:,1]

    if conn_type=='past':
        return tot[a_idxs>b_idxs].T
    elif conn_type=='pres':
        return tot[a_idxs==b_idxs].T
    elif conn_type=='fut':
        return tot[a_idxs<b_idxs].T
    else: 
        assert False


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
        return torch.tensor([]).to(torch.float32)
    else:
        if 'mosi' in gc['dataset'] or 'iemocap' in gc['dataset']: # front padded
            idx = (arr==0).all(dim=-1).to(torch.long).argmin()
            return arr[idx:]

        elif 'social' in gc['dataset']: # back padded
            # find idx of last zero element looking from back to front
            idx = (arr==0).all(dim=-1).to(torch.long).flip(dims=[0]).argmin()
            return arr[:-idx]
            
        else: 
            assert False, 'Only social, mosi, and iemocap are supported right now.  To add another dataset, break here and see whether front or back padded to seq len'

def get_loader_solograph(ds, dsname):
    # total_data = load_pk(dsname)
    # total_data = None
    # if total_data is None:
    print(f'Regenerating graphs for {dsname}')
    words = ds[:][0]
    covarep = ds[:][1]
    facet = ds[:][2]

    global gc
    if 'social' in gc['dataset']:
        q,a,inc=[torch.from_numpy(data[:]) for data in ds[0]]
        
        q = q.reshape(-1, q.shape[1]*q.shape[2], q.shape[-2], q.shape[-1]) # [888, 6, 12, 1, 25, 768] -> [888, 72, 25, 768]
        a = a.reshape(-1, a.shape[1]*a.shape[2], a.shape[-2], a.shape[-1]) # [888, 6, 12, 1, 25, 768] -> [888, 72, 25, 768]
        inc = inc.reshape(-1, inc.shape[1]*inc.shape[2], inc.shape[-2], inc.shape[-1]) # [888, 6, 12, 1, 25, 768] -> [888, 72, 25, 768]

        facet=torch.from_numpy(ds[1][:,:,:].transpose(1,0,2))
        words=torch.from_numpy(ds[2][:,:,:].transpose(1,0,2))
        covarep=torch.from_numpy(ds[3][:,:,:].transpose(1,0,2))
        gc['true_bs'] = gc['bs']*q.shape[1] # bs refers to number of videos processed at once, but each q-a-mods is a different graph

    total_data = []
    for i in tqdm(range(words.shape[0])):
        data = {
            'text': get_masked(words[i]),
            'audio': get_masked(covarep[i]),
            'video': get_masked(facet[i]),
        }

        if gc['zero_out_video']:
            data['video'][:]=0
        if gc['zero_out_audio']:
            data['audio'][:]=0
        if gc['zero_out_text']:
            data['text'][:]=0
        
        if sum([len(v) for v in data.values()]) == 0:
            continue
                
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

        for j in range(q.shape[1]): # for each of the 72 q-a pairs, make a new graph
            # Q-A connections
            _q = q[i,j][None,:,:] # [1,25,768]
            _a = a[i,j][None,:,:] # [1,25,768]
            _inc = inc[i,j][None,:,:] # [1,25,768]

            if np.random.random() < .5: # randomly flip if answer or incorrect is first; shouldn't matter in graph setup
                _a1 = _a
                _a2 = _inc
                a_idx = torch.Tensor([1,0]).to(torch.long)
                i_idx = torch.Tensor([0,1]).to(torch.long)

            else:
                _a1 = _inc
                _a2 = _a
                a_idx = torch.Tensor([0,1]).to(torch.long)
                i_idx = torch.Tensor([1,0]).to(torch.long)
            
            _as = torch.cat([_a1, _a2], dim=0)

            # Q/A - MOD
            for mod in mods:
                data['q', f'q_{mod}', mod] = torch.cat([torch.zeros(data[mod].shape[0])[None,:], torch.arange(data[mod].shape[0])[None,:]], dim=0).to(torch.long) # [ [0,0,...0], [0,1,...len(mod)]]
                data[mod, f'{mod}_q', 'q'] = torch.clone(data['q', f'q_{mod}', mod]).flip(dims=[0])

                data['a', f'a_{mod}', mod] = torch.cat([
                    torch.cat([torch.zeros(data[mod].shape[0])[None,:], torch.arange(data[mod].shape[0])[None,:]], dim=0).to(torch.long),
                    torch.cat([torch.ones(data[mod].shape[0])[None,:], torch.arange(data[mod].shape[0])[None,:]], dim=0).to(torch.long)
                ], dim=-1)
                data[mod, f'{mod}_a', 'a'] = torch.clone(data['a', f'a_{mod}', mod]).flip(dims=[0])

            # Q-A
            data['q', 'q_a', 'a'] = torch.Tensor([ [0,0], [0,1] ]).to(torch.long)
            data['a', 'a_q', 'q'] = torch.clone(data['q', 'q_a', 'a']).flip(dims=[0])

            # A-A
            data['a', 'a_a', 'a'] = torch.Tensor([[0,0,1,1],[0,1,0,1]]).to(torch.long)

            hetero_data = {
                **{k: {'x': v} for k,v in data.items() if 'idx' not in k and not isinstance(k, tuple)}, # just get data on mods
                **{k: {'edge_index': v} for k,v in data.items() if isinstance(k, tuple) },
                'q': {'x': _q},
                'a': {'x': _as},
                'a_idx': {'x': a_idx},
                'i_idx': {'x': i_idx},
                'ds_idx': len(total_data), # index of sample in underlying dataloader data
            }

            hetero_data = HeteroData(hetero_data) # different "sample" for each video-q-a graph
            
            # hetero_data = T.AddSelfLoops()(hetero_data) # todo: include this as a HP to see if it does anything!
            if 'social' not in gc['dataset']:
                hetero_data.y = ds[i][-1]
                hetero_data.id = ds.ids[i]
            
            total_data.append(hetero_data)
        
        if i == 12 and gc['test']:
            break
        
    # save_pk(dsname, total_data)

    # testing
    loader = DataLoader(total_data, batch_size=2, shuffle=False)
    if 'train' in dsname:
        batch = next(iter(loader))
        assert torch.all(batch['q', 'q_text', 'text']['edge_index'] == torch.Tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]]).to(torch.long)).item()
        assert torch.all(batch['text', 'text_q', 'q']['edge_index'] == torch.Tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]).to(torch.long)).item()

    loader = DataLoader(total_data, batch_size=1, shuffle=True)
    return loader


def get_loader(ds):
    words = ds[:][0]
    covarep = ds[:][1]
    facet = ds[:][2]

    if 'social' in gc['dataset']:
        q,a,inc=[torch.from_numpy(data[:]) for data in ds[0]]
        facet=torch.from_numpy(ds[1][:,:,:].transpose(1,0,2))
        words=torch.from_numpy(ds[2][:,:,:].transpose(1,0,2))
        covarep=torch.from_numpy(ds[3][:,:,:].transpose(1,0,2))

    total_data = []
    for i in range(words.shape[0]):
        data = {
            'text': get_masked(words[i]),
            'audio': get_masked(covarep[i]),
            'video': get_masked(facet[i]),
        }

        if gc['zero_out_video']:
            data['video'][:]=0
        if gc['zero_out_audio']:
            data['audio'][:]=0
        if gc['zero_out_text']:
            data['text'][:]=0
        
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
        # for mod in mods:
        #     assert isinstance(data[mod], torch.Tensor)
        #     for mod2 in [modx for modx in mods if modx != mod]:
        #         if (mod, 'fut', mod2) in data:
        #             assert (data[mod, 'fut', mod2].flip(dims=[0]) == data[mod2, 'past', mod]).all()
        #             assert isinstance(data[mod, 'fut', mod2], torch.Tensor) and isinstance(data[mod, 'past', mod2], torch.Tensor) and isinstance(data[mod, 'pres', mod2], torch.Tensor)

        hetero_data = {
            **hetero_data,
            **{k: {'edge_index': v} for k,v in data.items() if isinstance(k, tuple) }
        }
        if 'social' in gc['dataset']:
            if gc['graph_qa']:
                hetero_data = {
                    **hetero_data,
                    'q': {'x': q[i]},
                    'a': {'x': a[i]},
                    'inc': {'x': inc[i]},
                }

            hetero_data = {
                **hetero_data,
                'q': q[i],
                'a': a[i],
                'inc': inc[i],
                'vis': facet[i],
                'trs': words[i],
                'acc': covarep[i],
            }
        
            if gc['qa_strat']==1:
                hi=2

        hetero_data = HeteroData(hetero_data)
        
        # hetero_data = T.AddSelfLoops()(hetero_data) # todo: include this as a HP to see if it does anything!
        if 'social' not in gc['dataset']:
            hetero_data.y = ds[i][-1]
            hetero_data.id = ds.ids[i]
        
        total_data.append(hetero_data)

    loader = DataLoader(total_data, batch_size=gc['bs'], shuffle=True)
    return loader


class SocialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_lstm=mylstm.MyLSTM(768,50)
        self.a_lstm=mylstm.MyLSTM(768,50)
        
        self.judge = nn.Sequential(OrderedDict([
            ('fc0',   nn.Linear(214,25)),
            ('drop_1', nn.Dropout(p=gc['drop_1'])),
            ('sig0', nn.Sigmoid()),
            ('fc1',   nn.Linear(25,1)),
            ('drop_2', nn.Dropout(p=gc['drop_2'])),
            ('sig1', nn.Sigmoid())
        ]))

        self.hetero_gnn = HeteroGNN(gc['graph_conv_in_dim'], 1, gc['num_gat_layers'])

    def forward(self, batch):
        hetero_out = self.hetero_gnn(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        hetero_reshaped = hetero_out[:,None,:].expand(-1, 12*6, -1).reshape(-1, gc['graph_conv_in_dim'])
        hetero_normed = (hetero_reshaped - hetero_reshaped.mean(dim=-1)[:,None]) / (hetero_reshaped.std(dim=-1)[:,None] + 1e-6)

        q = batch.q.reshape(-1, 6, *batch.q.shape[1:])
        a = batch.a.reshape(-1, 6, *batch.a.shape[1:])
        inc = batch.inc.reshape(-1, 6, *batch.inc.shape[1:])
        
        q_rep=self.q_lstm.step(to_pytorch(flatten_qail(q)))[1][0][0,:,:]
        a_rep=self.a_lstm.step(to_pytorch(flatten_qail(a)))[1][0][0,:,:]
        i_rep=self.a_lstm.step(to_pytorch(flatten_qail(inc)))[1][0][0,:,:]

        correct=self.judge(torch.cat((q_rep,a_rep,i_rep,hetero_normed),1))
        incorrect=self.judge(torch.cat((q_rep,i_rep,a_rep,hetero_normed),1))

        return correct, incorrect

class MosiModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.hetero_gnn = HeteroGNN(hidden_channels, out_channels, num_layers)

        self.finalW = nn.Sequential(
            Linear(-1, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(p=gc['drop_1']),
            Linear(hidden_channels // 4, hidden_channels // 4),
            nn.Dropout(p=gc['drop_2']),
            nn.ReLU(),
            Linear(hidden_channels // 4, out_channels),
        )

    def forward(self, batch):
        hetero_out = self.hetero_gnn(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        return self.finalW(hetero_out).squeeze(axis=-1)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = gc['gat_conv_num_heads']
        
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in mods:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            # conv = HeteroConv({
            #     conn_type: GATv2Conv(gc['graph_conv_in_dim'], hidden_channels//self.heads, heads=self.heads)
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
                    gc['graph_conv_in_dim'], 
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

        self.pes = {k: PositionalEncoding(gc['graph_conv_in_dim']) for k in mods}

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
        return x

def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(train_loader, model, optimizer):
    total_loss, total_examples = 0,0
    y_trues = []
    y_preds = []
    model.train()
    if 'iemocap' in gc['dataset']:
        criterion = IEMOCAPInverseSampleCountCELoss()
        criterion.to(device)
    else: # mosi
        criterion = nn.SmoothL1Loss()
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
                out = model(data)

        optimizer.zero_grad()

        out = model(data)
        if 'iemocap' in gc['dataset']:
            loss = criterion(out.view(-1,2), data.y.view(-1))
            
        else:
            loss = criterion(out, data.y)
        
        if gc['use_loss_norm']:
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
    ids = []
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

        if 'aiEXnCPZubE_24' in data.id:
            a=2
        data = data.to(device)
        if 'iemocap' in gc['dataset']:
            data.y = data.y.reshape(-1,4)
        out = model(data)
        if 'iemocap' in gc['dataset']:
            loss = nn.CrossEntropyLoss()(out, data.y.argmax(-1)).item()
        else:
            loss = F.mse_loss(out, data.y)
            loss = loss / torch.abs(loss.detach()) # norm
            l += F.mse_loss(out, data.y, reduction='mean').item()

        y_true = data.y.detach().cpu().numpy()
        y_pred = out.detach().cpu().numpy()
        
        y_trues.extend(y_true)
        y_preds.extend(y_pred)
        ids.extend(data.id)

        del data
        del out
    
    # if valid:
    #     scheduler.step(mse)
    return l if l != 0 else loss, y_trues, y_preds

paths={}
# paths["QA_BERT_lastlayer_binarychoice"]="/work/awilf/Social-IQ/socialiq/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
paths["QA_BERT_lastlayer_binarychoice"]="/work/awilf/MTAG/deployed/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
paths["DENSENET161_1FPS"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_DENSENET161_1FPS'.csd"
paths["Transcript_Raw_Chunks_BERT"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT'.csd"
paths["Acoustic"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_COVAREP'.csd"
social_iq=mmdatasdk.mmdataset(paths)
social_iq.unify()
NUM_QS = 6
NUM_A_COMBS = 12

qa_conns = [
    ('text', 'text_q', 'q'),
    ('audio', 'audio_q', 'q'), 
    ('video', 'video_q', 'q'), 
    
    ('text', 'text_a', 'a'), 
    ('audio', 'audio_a', 'a'), 
    ('video', 'video_a', 'a'),

    # flipped above
    ('q', 'q_text','text'),
    ('q', 'q_audio','audio'), 
    ('q', 'q_video','video'), 
    
    ('a', 'a_text','text'), 
    ('a', 'a_audio','audio'), 
    ('a', 'a_video','video'),

    ('q', 'q_a', 'a'),
    ('a', 'a_q', 'q'),

    # a to i
    ('a', 'ai', 'a'),

    # self conns
    ('a', 'a_a', 'a'),
    ('q', 'q_q', 'q'),
]

def bds_to_conns(a,b): # a is batch_dict_1, b is batch_dict_2, return all combinations of the indices that share a batch dict
    '''
    e.g.
    a = torch.from_numpy(ar([0,0,1,1,2,2,3]))
    b = torch.from_numpy(ar([0,1,2,3]))
    '''
    b = b[:,None].expand(-1,a.shape[0])
    b_to_a = torch.vstack(torch.where(a[None,:]==b)) # top is b idxs, bot is a idxs
    return b_to_a

def interleave(a,b):
    '''
    two tensors of shape (n,...), (n,...), where ... is the same but doesn't matter what it is
    return an interleaved version of shape (2n, ...) of the form (a1,b1, b2,a2, b3,a3, a4,b4) where the order of the pairs is randomized.  return interleaved, a_idxs, b_idxs
    '''
    is_a_first = torch.randint(low=0,high=2, size=(b.shape[0],))

    # is_a_first = torch.Tensor([0,1,0,0,1,1])

    complement = (is_a_first + 1) % 2
    tot_idxs = torch.zeros(b.shape[0]*2, dtype=torch.long)
    tot_idxs[0::2] = is_a_first
    tot_idxs[1::2] = complement

    a_idxs = torch.where(tot_idxs==1)[0]
    b_idxs = torch.where(tot_idxs==0)[0]

    assert (a_idxs.shape[0] + b_idxs.shape[0]) == tot_idxs.shape[0]

    interleaved = torch.zeros((b.shape[0]*2, *b.shape[1:]), dtype=b.dtype)
    interleaved = interleaved.to(device)
    interleaved[a_idxs] = a
    interleaved[b_idxs] = b

    return interleaved, a_idxs, b_idxs

def get_mesh(a,b): # a is first set of indices, b is second
    # return torch.Tensor(np.array(np.meshgrid(a, b))).reshape(2,-1).to(torch.long)
    return torch.cat([elt.unsqueeze(0) for elt in torch.meshgrid(a,b)]).reshape(2,-1).to(torch.long)

def get_q_mod_edges(q, v, bi): # q is question indices, v is modality indices, bi is batch indices of modality
    # split original modality into batches
    idxs = torch.where(torch.diff(bi))[0]+1
    if idxs.shape[0] == 0:
        splits = [v]
    else:
        splits = torch.split(v, idxs)

    meshs = []
    for i, split in enumerate(splits):
        qs = q[NUM_QS*i:NUM_QS*(i+1)]
        meshs.append(get_mesh(split, qs))

    all_edges = (torch.cat(meshs, dim=-1)).to(torch.long)
    return all_edges

class GraphQA_HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = gc['gat_conv_num_heads']
        
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in mods:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.qa_convs = torch.nn.ModuleList()

        for i in range(num_layers):          

            # UNCOMMENT FOR PARAMETER SHARING
            mods_seen = {} # mapping from mod to the gatv3conv linear layer for it
            d = {}
            for conn_type in all_connections + qa_conns:
                mod_l, _, mod_r = conn_type

                lin_l = None if mod_l not in mods_seen else mods_seen[mod_l]
                lin_r = None if mod_r not in mods_seen else mods_seen[mod_r]

                _conv =  GATv3Conv(
                    lin_l,
                    lin_r,
                    gc['graph_conv_in_dim'], 
                    hidden_channels//self.heads,
                    heads=self.heads,
                    dropout=gc['drop_het'],
                )
                if mod_l not in mods_seen:
                    mods_seen[mod_l] = _conv.lin_l
                if mod_r not in mods_seen:
                    mods_seen[mod_r] = _conv.lin_r
                d[conn_type] = _conv
            
            conv = HeteroConv(d, aggr='mean')

            self.convs.append(conv)

            # conv = HeteroConv({
            #     conn_type: GATv2Conv(gc['graph_conv_in_dim'], hidden_channels//self.heads, heads=self.heads)
            #     for conn_type in qa_conns+all_connections
            # }, aggr='mean')
            # self.qa_convs.append(conv)

        self.pes = {k: PositionalEncoding(gc['graph_conv_in_dim']) for k in mods}

    def map_x(self, x, key):
        if key in mods:
            return self.lin_dict[key](x)
        # elif key == 'q':
        #     return self.q_lstm.step(x)
        # elif key == 'a':
        #     return self.a_lstm.step(x)
        else:
            assert False, key+' not an accepted key!'

    def forward(self, x_dict, edge_index_dict, batch_dict, q_rep, a_rep, i_rep, block):
        # linearly project modalities
        x_dict = { key: self.map_x(x, key) for key, x in x_dict.items() }
        
        # apply pe
        for m, v in x_dict.items(): # modality, tensor
            idxs = batch_dict[m]
            assert (idxs==(idxs.sort().values)).all()
            _, counts = torch.unique(idxs, return_counts=True)
            x_dict[m] = self.pes[m](v, counts)

        # # for testing
        # NUM_QS = 2
        # NUM_A_COMBS = 6
        # bdv = torch.Tensor([0,0,0,0,1,1])
        # batch_dict['v'] = bdv.to(device)
        # x_dict['v'] = torch.ones(6,10)
        # q_rep = torch.ones(2,2,10).reshape(-1,10)
        # a_rep = torch.ones(2,2,6,10).reshape(-1,10)
        # i_rep = torch.ones(2,2,6,10).reshape(-1,10)
        # a_rep = torch.cat((a_rep,i_rep),dim=0)
        # mods = ['v']
        ##
        
        x_dict['q'] = q_rep
        new_eid = {}
        bs = q_rep.shape[0] // NUM_QS
        a_idxs = torch.arange(NUM_QS*bs*NUM_A_COMBS*2)
        
        ## Q - mods
        # get batch dict
        # like [0,0...,0,1,1,...1,...32,32...32] where each batch idx repeats 6 times for the 6 questions
        bd_q = torch.arange(bs)[:,None].expand(-1,NUM_QS).reshape(-1)
        bd_q = bd_q.to(device)

        for mod in mods:
            new_eid[mod, f'{mod}_q', 'q'] = bds_to_conns(bd_q, batch_dict[mod])
            new_eid['q', f'q_{mod}', mod] = bds_to_conns(bd_q, batch_dict[mod]).flip(dims=[0])

        # create new edge index dict for qa to be added in at end
        ## BLOCK
        if block:
            x_dict['a'] = torch.cat((a_rep, i_rep),0)
            ## A / I - mods
            bd_a = torch.arange(bs)[:,None].expand(-1,NUM_QS*NUM_A_COMBS).reshape(-1)
            bd_a = torch.cat((bd_a, bd_a), 0) # for i
            bd_a = bd_a.to(device)
            ai_split = bs*NUM_A_COMBS*NUM_QS # partition idx between a and i

            if gc['use_mod_conn']:
                for mod in mods:
                    new_eid[mod, f'{mod}_a', 'a'] = bds_to_conns(bd_a, batch_dict[mod])
                    new_eid['a', f'a_{mod}', mod] = bds_to_conns(bd_a, batch_dict[mod]).flip(dims=[0])

            ## Q-A
            q_idxs = torch.arange(NUM_QS*bs)[:,None].expand(-1, NUM_A_COMBS).reshape(-1)
            q_idxs = torch.cat((q_idxs, q_idxs))

            if gc['use_qa_conn']:
                new_eid['q', 'q_a', 'a'] = torch.vstack([q_idxs, a_idxs])
                new_eid['a', 'a_q', 'q'] = torch.vstack([a_idxs, q_idxs])
            
            if gc['use_ai_conn']:
                new_eid['a', 'ai', 'a'] = torch.cat((torch.vstack([a_idxs[:ai_split], a_idxs[ai_split:]]), torch.vstack([a_idxs[ai_split:], a_idxs[:ai_split]])), dim=-1)
            
            if gc['use_qa_self_conn']:
                q_idxs_unexpanded = torch.arange(NUM_QS*bs)
                new_eid['q', 'q_q', 'q'] = torch.vstack([q_idxs_unexpanded, q_idxs_unexpanded])
                new_eid['a', 'a_a', 'a'] = torch.vstack([a_idxs, a_idxs])

        ##

        ## INTERLEAVE
        else:
            x_dict['a'] = a_rep
            ## A - mods
            bd_a = torch.arange(bs)[:,None].expand(-1,NUM_QS*NUM_A_COMBS*2).reshape(-1)
            bd_a = bd_a.to(device)
            if gc['use_mod_conn']:
                for mod in mods:
                    new_eid[mod, f'{mod}_a', 'a'] = bds_to_conns(bd_a, batch_dict[mod])
                    new_eid['a', f'a_{mod}', mod] = bds_to_conns(bd_a, batch_dict[mod]).flip(dims=[0])

            ## Q-A
            bd_qq = torch.arange(NUM_QS*bs) # treat q as "batch dict of questions", where q0 comes from question 0
            bd_aq = torch.arange(NUM_QS*bs)[:,None].expand(-1, NUM_A_COMBS*2).reshape(-1) # treat bd_qa as "batch dict of questions", where idxs 0:NUM_A_COMBS*2 come from q1, NUM_A_COMBS*2:2*NUM_A_COMBS*2 come from q2...etc
            if gc['use_qa_conn']:
                new_eid['q', 'q_a', 'a'] = bds_to_conns(bd_aq, bd_qq)
                new_eid['a', 'a_q', 'q'] = bds_to_conns(bd_aq, bd_qq).flip(dims=[0])

            if gc['use_ai_conn']:
                new_eid['a', 'ai', 'a'] = torch.cat( (torch.arange(a_rep.shape[0]).reshape(-1,2).t(), torch.arange(a_rep.shape[0]).reshape(-1,2).t().flip(dims=[0])), dim=-1)
                
            if gc['use_qa_self_conn']:
                q_idxs_unexpanded = torch.arange(NUM_QS*bs)
                new_eid['q', 'q_q', 'q'] = torch.vstack([q_idxs_unexpanded, q_idxs_unexpanded])
                new_eid['a', 'a_a', 'a'] = torch.vstack([a_idxs, a_idxs])


        edge_index_dict = {
            **edge_index_dict,
            **new_eid,
        }
        # move all to cuda
        for k,v in edge_index_dict.items():
            if v.device.type=='cpu':
                edge_index_dict[k] = edge_index_dict[k].to(device)

        for conv in self.convs:
            x_dict, _ = conv(x_dict , edge_index_dict, return_attention_weights_dict={elt: True for elt in qa_conns+all_connections})
            x_dict = {key: x[0].relu() for key, x in x_dict.items()}

        # readout: avg nodes (no pruning yet!)
        q_out = x_dict['q'].reshape(-1,NUM_QS,gc['graph_conv_in_dim'])[:,:,None,:].expand(-1,-1,NUM_A_COMBS,-1).reshape(-1, gc['graph_conv_in_dim']) # (bs*6*12, 64)
        
        if block:
            ai_split = bs*NUM_A_COMBS*NUM_QS # partition idx between a and i
            a_out = x_dict['a'][:ai_split] # (bs*6*12, 64)
            i_out = x_dict['a'][ai_split:] # (bs*6*12, 64)
            return q_out, a_out, i_out

        else:
            a_out = x_dict['a']
            return q_out, a_out

class GraphQA_SocialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_lstm = mylstm.MyLSTM(768,gc['graph_conv_in_dim'])
        self.a_lstm = mylstm.MyLSTM(768,gc['graph_conv_in_dim'])

        self.judge = nn.Sequential(OrderedDict([
            ('fc0',   nn.Linear(3*gc['graph_conv_in_dim'],25)),
            ('drop_1', nn.Dropout(p=gc['drop_1'])),
            ('sig0', nn.Sigmoid()),
            ('fc1',   nn.Linear(25,1)),
            ('drop_2', nn.Dropout(p=gc['drop_2'])),
            ('sig1', nn.Sigmoid())
        ]))

        self.hetero_gnn = GraphQA_HeteroGNN(gc['graph_conv_in_dim'], 1, gc['num_gat_layers'])

    def forward_block(self, batch):
        q = batch.q[:,0,0,:,:]
        q_rep=self.q_lstm.step(q.transpose(1,0))[1][0][0,:,:]

        
        a = batch.a.reshape(-1, 6, *batch.a.shape[1:])
        inc = batch.inc.reshape(-1, 6, *batch.inc.shape[1:])
        
        a_rep=self.a_lstm.step(to_pytorch(flatten_qail(a)))[1][0][0,:,:]
        i_rep=self.a_lstm.step(to_pytorch(flatten_qail(inc)))[1][0][0,:,:]

        q_out, a_out, i_out = self.hetero_gnn(batch.x_dict, batch.edge_index_dict, batch.batch_dict, q_rep, a_rep, i_rep, block=True)

        correct = self.judge(torch.cat((q_out, a_out,i_out),1))
        incorrect = self.judge(torch.cat((q_out, i_out,a_out),1))
        return correct, incorrect

    def forward_inter(self, batch, ai_rep, a_idxs, i_idxs):
        q = batch.q[:,0,0,:,:]
        q_rep=self.q_lstm.step(q.transpose(1,0))[1][0][0,:,:]

        a_rep=self.a_lstm.step(ai_rep)[1][0][0,:,:]

        q_out, a_out = self.hetero_gnn(batch.x_dict, batch.edge_index_dict, batch.batch_dict, q_rep, a_rep, i_rep=None, block=False)
        i_out = a_out[i_idxs]
        a_out = a_out[a_idxs]

        correct = self.judge(torch.cat((q_out, a_out, i_out),1))
        incorrect = self.judge(torch.cat((q_out, i_out, a_out),1))
        return correct, incorrect
        
non_mod_nodes = ['q', 'a', 'a_idx', 'i_idx', 'agg']

class Solograph_HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = gc['gat_conv_num_heads']

        assert self.hidden_channels % self.heads == 0, 'Hidden channels must be divisible by number of heads'

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in mods:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            conv = HeteroConv({
                conn_type: GATv2Conv(gc['graph_conv_in_dim'], hidden_channels//self.heads, heads=self.heads, dropout=gc['drop_het'])
                for conn_type in all_connections + qa_conns
            }, aggr='mean')
            

            # UNCOMMENT FOR PARAMETER SHARING
            # mods_seen = {} # mapping from mod to the gatv3conv linear layer for it
            # d = {}
            # for conn_type in all_connections + qa_conns:
            #     mod_l, _, mod_r = conn_type

            #     lin_l = None if mod_l not in mods_seen else mods_seen[mod_l]
            #     lin_r = None if mod_r not in mods_seen else mods_seen[mod_r]

            #     _conv =  GATv3Conv(
            #         lin_l,
            #         lin_r,
            #         gc['graph_conv_in_dim'], 
            #         hidden_channels//self.heads,
            #         heads=self.heads,
            #         dropout=gc['drop_het'],
            #     )
            #     if mod_l not in mods_seen:
            #         mods_seen[mod_l] = _conv.lin_l
            #     if mod_r not in mods_seen:
            #         mods_seen[mod_r] = _conv.lin_r
            #     d[conn_type] = _conv
            
            # conv = HeteroConv(d, aggr='mean')

            self.convs.append(conv)

        self.pes = {k: PositionalEncoding(gc['graph_conv_in_dim']) for k in mods}

    def forward(self, x_dict, edge_index_dict, batch_dict, ds_idx):
        mod_dict = {k: v for k,v in x_dict.items() if k not in non_mod_nodes}
        qa_dict = {k: v for k,v in x_dict.items() if k in non_mod_nodes}
        mod_dict = {key: self.lin_dict[key](x) for key, x in mod_dict.items()}

        # apply pe
        for m, v in mod_dict.items(): # modality, tensor
            idxs = batch_dict[m]
            assert (idxs==(idxs.sort().values)).all()
            _, counts = torch.unique(idxs, return_counts=True)
            mod_dict[m] = self.pes[m](v, counts)

        x_dict = {
            **mod_dict,
            **qa_dict,
        }
        for conv in self.convs:
            x_dict, edge_types = conv(x_dict, edge_index_dict, return_attention_weights_dict={elt: True for elt in all_connections+qa_conns})

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

        new_attn_dict = {}
        for mod in mods:
            new_attn_dict = {**new_attn_dict, **attn_dict[mod]}
        attn_dict = new_attn_dict

        # update dataloader
        for mod_con in mod_conns:
            try:
                edge_weight = attn_dict[mod_con]['edge_weight'].mean(dim=-1)
                edge_index = attn_dict[mod_con]['edge_index']
                idxs = edge_weight.argsort(descending=True)
                num_nodes_keep = int(gc['prune_keep_p']*idxs.shape[0])
                idxs = idxs[:num_nodes_keep]
                pruned_edge_index = edge_index[:,idxs]
                pruned_edge_index = pruned_edge_index.cpu()
                loader.dataset[ds_idx[0]][mod_con]['edge_index'] = pruned_edge_index
            except:
                print(f'Failed because {mod_con} was not in attn_dict for elt ds_idx: {ds_idx}')
                hi=2

        # get mean scene rep
        if gc['scene_mean']:
            x = torch.cat([v for k,v in x_dict.items() if k not in non_mod_nodes], axis=0)
            batch_dicts = torch.cat([v for k,v in batch_dict.items() if k not in non_mod_nodes], axis=0)
            x = scatter_mean(x, batch_dicts, dim=0)
            scene_rep = x

            return x_dict['q'], x_dict['a'], scene_rep
        else:
            return x_dict['q'], x_dict['a']


def debug_mem():
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    r = torch.cuda.memory_reserved(0) / 1e9
    a = torch.cuda.memory_allocated(0) / 1e9
    f = r-a  # free inside reserved
    print('-- Debug Mem --')
    print(f'Allocated:{a:.4f}')
    print(f'Reserved:{r:.4f}')
    print(f'Free:{f:.4f}')
    print('-- --')

class Solograph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_lstm = mylstm.MyLSTM(768,gc['graph_conv_in_dim'])
        self.a_lstm = mylstm.MyLSTM(768,gc['graph_conv_in_dim'])

        lin_width = 4 if gc['scene_mean'] else 3
        self.judge = nn.Sequential(OrderedDict([
            ('fc0',   nn.Linear(lin_width*gc['graph_conv_in_dim'],25)),
            ('drop_1', nn.Dropout(p=gc['drop_1'])),
            ('sig0', nn.Sigmoid()),
            ('fc1',   nn.Linear(25,1)),
            ('drop_2', nn.Dropout(p=gc['drop_2'])),
            ('sig1', nn.Sigmoid())
        ]))

        self.hetero_gnn = Solograph_HeteroGNN(gc['graph_conv_in_dim'], 1, gc['num_gat_layers'])

    def forward(self, batch):
        x_dict = batch.x_dict
        x_dict['q'] = self.q_lstm.step(batch['q']['x'].transpose(1,0))[1][0][0,:,:] # input to q_lstm must be of shape ([25, num_qs, 768])
        x_dict['a'] = self.a_lstm.step(batch['a']['x'].transpose(1,0))[1][0][0,:,:] # input to a_lstm must be of shape ([25, num_qs, 768])

        a_idx, i_idx = x_dict['a_idx'], x_dict['i_idx']
        del x_dict['a_idx'] # should not be used by heterognn
        del x_dict['i_idx']

        # assert torch.all(torch.cat([a_idx[None,:], i_idxs[None,:]], dim=0).sum(dim=0) == 1).item()
        if gc['scene_mean']:
            q_out, a_out, scene_rep = self.hetero_gnn(x_dict, batch.edge_index_dict, batch.batch_dict, batch['ds_idx'])

            a = a_out[torch.where(a_idx)[0]]
            inc = a_out[torch.where(i_idx)[0]]

            correct = self.judge(torch.cat((q_out, a, inc, scene_rep), 1))
            incorrect = self.judge(torch.cat((q_out, inc, a, scene_rep), 1))
        
        else:
            q_out, a_out = self.hetero_gnn(x_dict, batch.edge_index_dict, batch.batch_dict, batch['ds_idx'])

            a = a_out[torch.where(a_idx)[0]]
            inc = a_out[torch.where(i_idx)[0]]

            correct = self.judge(torch.cat((q_out, a, inc), 1))
            incorrect = self.judge(torch.cat((q_out, inc, a), 1))

        return correct, incorrect


def get_model_out(batch, block, model, split):
    if gc['solograph'] or not gc['graph_qa']:
        return model(batch)
    if block:
        if gc[f'flip_{split}_order']:
            if np.random.random() < .5: # flip order of answer and incorrect in testing to make sure model isn't learning something fishy
                batch.a, batch.inc = batch.inc, batch.a
                incorrect, correct = model.forward_block(batch)
            else:
                correct, incorrect = model.forward_block(batch)
        else:
            correct, incorrect = model.forward_block(batch)
        
        return correct, incorrect
        
    else:
        a = batch.a.reshape(-1, 6, *batch.a.shape[1:])
        inc = batch.inc.reshape(-1, 6, *batch.inc.shape[1:])
        
        a = to_pytorch(flatten_qail(a))
        inc = to_pytorch(flatten_qail(inc))

        a = a.transpose(1,0)
        inc = inc.transpose(1,0)
        interleaved, a_idxs, i_idxs = interleave(a,inc)
        interleaved = interleaved.transpose(1,0)
        ai_rep = interleaved
        return model.forward_inter(batch, ai_rep, a_idxs, i_idxs)

train_loader, dev_loader = None, None # used to cache data preprocessing across trials
loader = None # used as global for pruning
def train_model_social(optimizer, use_gnn=True, exclude_vision=False, exclude_audio=False, exclude_text=False, average_mha=False, num_gat_layers=1, lr_scheduler=None, reduce_on_plateau_lr_scheduler_patience=None, reduce_on_plateau_lr_scheduler_threshold=None, multi_step_lr_scheduler_milestones=None, exponential_lr_scheduler_gamma=None, use_pe=False, use_prune=False):
    global train_loader, dev_loader, loader
    
    if train_loader is None: # cache train and dev loader so skip data loading in multiple iterations
        print('Building loaders for social')
        trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
        #This video has some issues in training set
        bads=['f5NJQiY9AuY','aHBLOkfJSYI']
        folds=[trk,dek]
        for bad in bads:
            for fold in folds:
                try:
                    fold.remove(bad)
                except:
                    pass

        preloaded_train=process_data(trk, 'train')
        preloaded_dev=process_data(dek, 'dev')
        replace_inf(preloaded_train[3])
        replace_inf(preloaded_dev[3])

        if gc['solograph']:
            train_loader = get_loader_solograph(preloaded_train, 'social_train')
            dev_loader = get_loader_solograph(preloaded_dev, 'social_dev')
        else:
            train_loader = get_loader(preloaded_train)
            dev_loader = get_loader(preloaded_dev)
        
        del preloaded_train
        del preloaded_dev

    #Initializing parameter optimizer
    if gc['solograph']:
        model = Solograph()
    elif gc['graph_qa']:
        model = GraphQA_SocialModel()
    else:
        model = SocialModel()
    
    model = model.to(device)
    params= list(model.q_lstm.parameters())+list(model.a_lstm.parameters())+list(model.judge.parameters())
    optimizer=optim.Adam(params,lr=gc['global_lr'])

    # graph optimizer
    graph_optimizer = torch.optim.AdamW(
        model.hetero_gnn.parameters(),
        lr=gc['global_lr'],
        weight_decay=gc['weight_decay']
    )

    print('Training...')
    metrics = {
        'train_acc_best':  0,
        'train_accs': [],
        'train_losses': [],

        'val_acc_best':  0,
        'val_accs':  [],
        'val_losses': [],
    }

    # epochs_since_new_max = 0 # early stopping
    for i in range(gc['epochs']):
        # if epochs_since_new_max > gc['early_stopping_patience'] and i > 15: # often has a slow start
        #     break
        print ("Epoch %d"%i)
        model.train()
        loader = train_loader
        train_losses, train_accs = [],[]

        train_block = gc['train_block']
        for batch_i, batch in enumerate(tqdm(train_loader)):
            batch = batch.to(device)

            skip = False
            for mod in mods:
                if len(batch[mod]['x'].shape) == 1 or batch[mod]['x'].shape[0] in [0,1]:
                    skip = True
            if skip:
                continue
                
            if batch_i == 0:
                with torch.no_grad():  # Initialize lazy modules.
                    correct, incorrect = model(batch)
                    del correct
                    del incorrect
                    torch.cuda.empty_cache()
            
            cont = False
            for mod in mods:
                if not np.any([mod in elt for elt in batch.edge_index_dict.keys()]):
                    print(mod, 'dropped from train loader!')
                    cont = True
            if cont:
                continue
            
            correct, incorrect = model(batch)

            correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
            incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()

            optimizer.zero_grad()
            graph_optimizer.zero_grad()

            loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean))
            loss.backward()
            
            optimizer.step()
            graph_optimizer.step()

            train_losses.append(loss.cpu().detach().numpy())
            train_accs.append(calc_accuracy(correct,incorrect))
            torch.cuda.empty_cache()

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        metrics['train_accs'].append(train_acc)
        metrics['train_losses'].append(train_loss)
        print (f"Train Acc: {train_acc:.4f}")
        print(f'Train loss:{train_loss:.4f}')

        metrics['train_acc_best'] = max(train_acc, metrics['train_acc_best'])

        val_losses, val_accs = [], []
        model.eval()
        loader = dev_loader
        test_block = gc['test_block']
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(device)
                
                skip = False
                for mod in mods:
                    if len(batch[mod]['x'].shape) == 1 or batch[mod]['x'].shape[0]==0:
                        skip = True
                if skip:
                    continue

                correct, incorrect = model(batch)
                
                correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
                incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
                
                acc = calc_accuracy(correct,incorrect)
                val_accs.append(acc)

                loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean)).cpu().detach().numpy()
                val_losses.append(loss)
                
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            metrics['val_accs'].append(val_acc)
            metrics['val_losses'].append(val_loss)
            print (f"Dev Acc: {val_acc:.4f}")
            print(f'Dev loss: {val_loss:.4f}')

            # epochs_since_new_max += 1
            # if val_acc > metrics['val_acc_best']:
            #     epochs_since_new_max = 0
            #     metrics['val_acc_best'] = val_acc

    print('Best dev acc:', metrics['val_acc_best'])
    print('Model parameters:', count_params(model))
    metrics['model_params'] = count_params(model)
    return metrics

trk,dek,preloaded_train,preloaded_dev = None, None, None, None
def train_social_baseline(optimizer, use_gnn=True, exclude_vision=False, exclude_audio=False, exclude_text=False, average_mha=False, num_gat_layers=1, lr_scheduler=None, reduce_on_plateau_lr_scheduler_patience=None, reduce_on_plateau_lr_scheduler_threshold=None, multi_step_lr_scheduler_milestones=None, exponential_lr_scheduler_gamma=None, use_pe=False, use_prune=False):
    #if you have enough RAM, specify this as True - speeds things up ;)
    global trk,dek,preloaded_train,preloaded_dev
    bs=32
    if trk is None:
        print('Loading data...')
        trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
        #This video has some issues in training set
        bads=['f5NJQiY9AuY','aHBLOkfJSYI']
        folds=[trk,dek]
        for bad in bads:
            for fold in folds:
                try:
                    fold.remove(bad)
                except:
                    pass

        preloaded_train=process_data(trk, 'train')
        preloaded_dev=process_data(dek, 'dev')
        replace_inf(preloaded_train[3])
        replace_inf(preloaded_dev[3])
    
    q_lstm,a_lstm,t_lstm,v_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn=init_tensor_mfn_modules()
    judge=get_judge().cuda()

    #Initializing parameter optimizer
    params=	list(q_lstm.parameters())+list(a_lstm.parameters())+list(judge.parameters())+\
        list(t_lstm.parameters())+list(v_lstm.parameters())+list(ac_lstm.parameters())+\
        list(mfn_mem.parameters())+list(mfn_delta1.parameters())+list(mfn_delta2.parameters())+list(mfn_tfn.linear_layer.parameters())

    optimizer=optim.Adam(params,lr=gc['global_lr'])

    metrics = {
        'all_train_accs': [],
        'all_train_losses': [],
        'all_dev_accs': [],
        'all_dev_losses': [],
    }

    for i in range(gc['epochs']):
        print ("Epoch %d"%i)
        losses=[]
        accs=[]
        ds_size=len(trk)
        for j in range(int(ds_size/bs)+1):

            this_trk=[j*bs,(j+1)*bs]

            q_rep,a_rep,i_rep,v_rep,t_rep,ac_rep,mfn_rep=feed_forward(this_trk,q_lstm,a_lstm,v_lstm,t_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn,preloaded_train)
                
            real_bs=float(q_rep.shape[0])

            correct=judge(torch.cat((q_rep,a_rep,i_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
            incorrect=judge(torch.cat((q_rep,i_rep,a_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
    
            correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
            incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
            
            optimizer.zero_grad()
            loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean))
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            accs.append(calc_accuracy(correct,incorrect))
            
        print ("Loss %f",numpy.array(losses,dtype="float32").mean())
        print ("Accs %f",numpy.array(accs,dtype="float32").mean())
        metrics['all_train_accs'].append(numpy.array(accs,dtype="float32").mean())
        metrics['all_train_losses'].append(numpy.array(losses,dtype="float32").mean())

        with torch.no_grad():
            _losses,_accs=[],[]
            ds_size=len(dek)
            for j in range(int(ds_size/bs)):

                this_dek=[j*bs,(j+1)*bs]

                q_rep,a_rep,i_rep,v_rep,t_rep,ac_rep,mfn_rep=feed_forward(this_dek,q_lstm,a_lstm,v_lstm,t_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn,preloaded_dev)

                real_bs=float(q_rep.shape[0])

                correct=judge(torch.cat((q_rep,a_rep,i_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
                incorrect=judge(torch.cat((q_rep,i_rep,a_rep,t_rep,v_rep,ac_rep,mfn_rep),1))

                correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
                incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
                loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean))

                _accs.append(calc_accuracy(correct,incorrect))
                _losses.append(loss.cpu().detach().numpy())
            
        print ("Dev Accs %f",numpy.array(_accs,dtype="float32").mean())
        print ("Dev Losses %f",numpy.array(_losses,dtype="float32").mean())
        print ("-----------")
        metrics['all_dev_accs'].append(numpy.array(_accs,dtype="float32").mean())
        metrics['all_dev_losses'].append(numpy.array(_losses,dtype="float32").mean())
    
    metrics['model_params'] = sum(p.numel() for p in params if p.requires_grad)
    return metrics

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

    out_channels = 8 if 'iemocap' in gc['dataset'] else 1
    model = MosiModel(gc['graph_conv_in_dim'], out_channels, gc['num_gat_layers'])
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

    best_valid_ie_f1s = {emo: 0 for emo in ie_emos}
    best_test_ie_f1s = {emo: 0 for emo in ie_emos}
    best = { 'mae': 0, 'corr': 0, 'acc_2': 0, 'acc_7': 0, 'ex_zero_acc': 0, 'f1_raven': 0, 'f1_mult': 0, }
    valid_best = { 'mae': 0, 'corr': 0, 'acc_2': 0, 'acc_7': 0, 'ex_zero_acc': 0, 'f1_raven': 0, 'f1_mult': 0, }


    for epoch in range(gc['epochs']):
        loss, y_trues_train, y_preds_train = train(train_loader, model, optimizer)
        train_res = eval_fn('train', y_preds_train, y_trues_train)

        valid_loss, y_trues_valid, y_preds_valid = test(valid_loader, model, scheduler, valid=True)
        valid_res = eval_fn('valid', y_preds_valid, y_trues_valid)
        
        if epoch == 10:
            a=2
        test_loss, y_trues_test, y_preds_test = test(test_loader, model, scheduler, valid=False)
        test_res = eval_fn('test', y_preds_test, y_trues_test)

        if 'iemocap' in gc['dataset']:
            for emo in ie_emos:
                if valid_res['f1'][emo] > best_valid_ie_f1s[emo]:
                    best_valid_ie_f1s[emo] = valid_res['f1'][emo]
                    best_test_ie_f1s[emo] = test_res['f1'][emo]
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f} Valid: '+str([f'{emo}: {valid_res["f1"][emo]:.4f} ' for emo in ie_emos]), 'Test: '+str([f'{emo}: {test_res["f1"][emo]:.4f} ' for emo in ie_emos]))

        else: # mosi/mosei
            if valid_res['acc_2'] > valid_best['acc_2']:
                for k in valid_best.keys():
                    valid_best[k] = valid_res[k]

                for k in best.keys():
                    best[k] = test_res[k]

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_res["acc_2"]:.4f}, ' f'Valid: {valid_res["acc_2"]:.4f}, Test: {test_res["acc_2"]:.4f}')
        
    if 'iemocap' in gc['dataset']:
        print('\n Test f1s at valid best:', best_test_ie_f1s)
        print('\n Valid f1s at valid best:', best_valid_ie_f1s)
        return best_test_ie_f1s
    else:
        print(f'\nBest test acc: {best["acc_2"]:.4f}')
        return {k: float(v) for k,v in best.items()}

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
    assert gc['task'] in ['mosi', 'mosei', 'mosi_unaligned', 'mosei_unaligned', 'iemocap', 'iemocap_unaligned', 'social_unaligned'], "Unsupported task. Should be either mosi or mosei"

    gc['data_path'] = gc['dataroot']
    gc['dataset'] = gc['task']

    if not gc['eval']:
        start_time = time.time()
        
        if gc['social_baseline']:
            train_fn = train_social_baseline
        else:
            train_fn = train_model if 'social' not in gc['dataset'] else train_model_social
        
        all_results = []
        for trial in range(gc['trials']):
            print(f'\nTrial {trial}')
            util.set_seed(SEEDS[trial])
            best_results = train_fn(gc['optimizer'],
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
            all_results.append(best_results)

        all_results = {k: [dic[k] for dic in all_results] for k in all_results[0].keys()}
        all_results['model_params'] = all_results['model_params'][0]

        elapsed_time = time.time() - start_time
        out_dir = join(gc['out_dir'])
        mkdirp(out_dir)

        save_json(join(out_dir, 'results.json'), all_results)
    
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
