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
from consts import GlobalConsts as gc
from model import NetMTGATAverageUnalignedConcatMHA
from dataset.MOSEI_dataset import MoseiDataset
from dataset.MOSEI_dataset_unaligned import MoseiDatasetUnaligned
from dataset.MOSI_dataset import MosiDataset
from dataset.MOSI_dataset_unaligned import MosiDatasetUnaligned
from dataset.IEMOCAP_dataset import IemocapDatasetUnaligned, IemocapDataset
import logging
import util
import pathlib

from alex_utils import *
import standard_grid


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
    for emo_ind, em in enumerate(gc.best.iemocap_emos):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        # if epoch != None and epoch % 5 == 0 and split == 'test':
        #     # import ipdb
        #     # ipdb.set_trace()
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        emos_f1[em] = f1
        acc = accuracy_score(test_truth_i, test_preds_i)
        emos_acc[em] = acc
        logging.info("\t%s %s F1 Score: %f" % (split, gc.best.iemocap_emos[emo_ind], f1))
        logging.info("\t%s %s Accuracy: %f" % (split, gc.best.iemocap_emos[emo_ind], acc))
    return emos_f1, emos_acc


def eval_mosi_mosei(split, output_all, label_all):
    # The length of output_all / label_all is the number
    # of samples within that split
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

    logging.info("\t%s mean error: %f" % (split, mae))
    logging.info("\t%s correlation: %f" % (split, cor))
    logging.info("\t%s accuracy: %f" % (split, acc))
    logging.info("\t%s accuracy 7: %f" % (split, acc_7))
    logging.info("\t%s exclude zero accuracy: %f" % (split, ex_zero_acc))
    # left and right refers to left side / right side value in Table 1 of https://arxiv.org/pdf/1911.09826.pdf
    logging.info("\t%s F1 score (raven): %f " % (split, f1_raven))  # includes zeros, Non-negative VS. Negative
    logging.info("\t%s F1 score (mult): %f " % (split, f1_mult))  # exclude zeros, Positive VS. Negative
    return mae, cor, acc, ex_zero_acc, acc_7, f1_mfn, f1_raven, f1_mult


def logSummary():
    logging.info("best epoch: %d" % gc.best.best_epoch)
    if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
        for split in ["test", "valid", "test_at_valid_max"]:
            for em in gc.best.iemocap_emos:
                print("highest %s %s F1: %f" % (split, em, gc.best.max_iemocap_f1[split][em]))
                print("highest %s %s accuracy: %f" % (split, em, gc.best.max_iemocap_acc[split][em]))
    else:
        logging.info("lowest training MAE: %f" % gc.best.min_train_mae)

        logging.info("best validation epoch: %f" % gc.best.best_val_epoch)

        logging.info("best validation epoch lr: {}".format(gc.best.best_val_epoch_lr))
        logging.info("lowest validation MAE: %f" % gc.best.min_valid_mae)

        logging.info("highest validation correlation: %f" % gc.best.max_valid_cor)
        logging.info("highest validation accuracy: %f" % gc.best.max_valid_acc)
        logging.info("highest validation exclude zero accuracy: %f" % gc.best.max_valid_ex_zero_acc)
        logging.info("highest validation accuracy 7: %f" % gc.best.max_test_acc_7)
        logging.info("highest validation F1 score (raven): %f" % gc.best.max_valid_f1_raven)
        logging.info("highest validation F1 score (mfn): %f" % gc.best.max_valid_f1_mfn)
        logging.info("highest validation F1 score (mult): %f" % gc.best.max_valid_f1_mult)

        for k, v in gc.best.checkpoints_val_mae.items():
            logging.info('checkpoints {} val mae {}'.format(k, v))

def summary_to_dict():
    results = {}

    if gc.dataset == "iemocap_unaligned" or gc.dataset == "iemocap":
        for split in ["valid"]:
            for em in gc.best.iemocap_emos:
                results[f"highest {split} {em} epoch"] = gc.best.best_iemocap_epoch[split][em]
                results[f"highest {split} {em} F1"] = gc.best.max_iemocap_f1[split][em]
                results[f"highest {split} {em} accuracy"] = gc.best.max_iemocap_acc[split][em]
    else:

        results["lowest training MAE"] = gc.best.min_train_mae
        results["lowest validation MAE"] = gc.best.min_valid_mae

        results["best validation epoch"] = gc.best.best_val_epoch
        results["best validation epoch lr"] = gc.best.best_val_epoch_lr
        results["lowest validation MAE"] = gc.best.min_valid_mae

        results["highest validation correlation"] = gc.best.max_valid_cor
        results["highest validation accuracy"] = gc.best.max_valid_acc
        results["highest validation exclude zero accuracy"] = gc.best.max_valid_ex_zero_acc
        results["highest validation accuracy 7"] = gc.best.max_valid_acc_7
        results["highest validation F1 score (raven)"] = gc.best.max_valid_f1_raven
        results["highest validation F1 score (mfn)"] = gc.best.max_valid_f1_mfn
        results["highest validation F1 score (mult)"] = gc.best.max_valid_f1_mult


        for k, v in gc.best.checkpoints_val_mae.items():
            results['checkpoints {} val mae'.format(k)] = v


        for k, v in gc.best.checkpoints_val_ex_0_acc.items():
            results['checkpoints {} val ex zero acc'.format(k)] = v

    return results


import random
def set_seed(my_seed):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)

set_seed(0)
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

from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

# get all connection types for declaring heteroconv later
mods = ['text', 'audio', 'video']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conn_types = ['past', 'pres', 'fut']
all_connections = []
for mod in mods:
    for mod2 in mods:
        for conn_type in conn_types:
            all_connections.append((mod, conn_type, mod2))


from graph_builder import construct_time_aware_dynamic_graph, build_time_aware_dynamic_graph_uni_modal, build_time_aware_dynamic_graph_cross_modal
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
            ret = build_time_aware_dynamic_graph_uni_modal(data[f'{mod}_idx'],[], [], 0, all_to_all=gc.config['use_all_to_all'], time_aware=True, type_aware=True)
            
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
        
        hetero_data = T.AddSelfLoops()(hetero_data) # todo: include this as a HP to see if it does anything!
        hetero_data.y = ds[i][-1]
        total_data.append(hetero_data)

    loader = DataLoader(total_data, batch_size=gc.config['batch_size'], shuffle=True)
    # loader = DataLoader(total_data, batch_size=2)
    # batch = next(iter(loader))
    return loader

from torch_geometric.nn import Linear
    
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from torch_scatter import scatter_mean
import torch.nn.functional as F
from gatv3conv import GATv3Conv

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

            # attn_dict = {
            #     k: {
            #         edge_type: {
            #             'edge_idxs': edge_idxs,
            #             'attn_weights': attn_weights,
            #         }
            #         for edge_type, (edge_idxs, attn_weights) in zip(edge_types[k], v[1])
            #     } 
            #     for k, v in x_dict.items()
            # }
            
            x_dict = {key: x[0].relu() for key, x in x_dict.items()}

        # readout: avg nodes (no pruning yet!)
        x = torch.cat([v for v in x_dict.values()], axis=0)
        batch_dicts = torch.cat([v for v in batch_dict.values()], axis=0)
        x = scatter_mean(x,batch_dicts, dim=0)
        return self.finalW(x).squeeze(axis=-1)

def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

debug = []

def train(train_loader, model, optimizer):
    total_loss, total_examples = 0,0
    global debug
    accs = []
    maes = []
    epoch_debug = []
    model.train()
    trn = enumerate(train_loader)
    for batch_i, data in tqdm(trn): # need index to prune edges
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
        loss = F.mse_loss(out, data.y)
        
        # norm
        loss = loss / torch.abs(loss.detach())

        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        total_examples += data.num_graphs

        y_true = (data.y >= 0).detach().cpu().numpy()
        y_pred = (out >= 0).detach().cpu().numpy()
        accs.append(accuracy_score(y_true, y_pred))

        maes.append(torch.nn.L1Loss()((out >= 0).to(torch.float).detach().cpu(),(data.y >= 0).to(torch.float).detach().cpu()))

        del loss
        del out
        epoch_debug.append([data.num_nodes, data.num_edges, torch.cuda.memory_allocated(0)])
        del data

    debug.append(epoch_debug)
    torch.cuda.empty_cache()
    return total_loss / total_examples, float(np.mean(accs)), float(np.mean(maes))

@torch.no_grad()
def test(loader, model, scheduler, valid):
    mse = []
    maes = []
    accs = []
    model.eval()

    mse = 0.0
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
        mse += F.mse_loss(out, data.y, reduction='mean').item()
        
        y_true = (data.y >= 0).detach().cpu().numpy()
        y_pred = (out >= 0).detach().cpu().numpy()
        accs.append(accuracy_score(y_true, y_pred))
        maes.append(torch.nn.L1Loss()((out >= 0).to(torch.float).detach().cpu(),(data.y >= 0).to(torch.float).detach().cpu()))

        del data
        del out
    
    # if valid:
    #     scheduler.step(mse)
    return mse, float(np.mean(accs)), float(np.mean(maes))
    
    
import gc as g
def train_model(optimizer, use_gnn=True, exclude_vision=False, exclude_audio=False, exclude_text=False, average_mha=False, num_gat_layers=1, lr_scheduler=None, reduce_on_plateau_lr_scheduler_patience=None, reduce_on_plateau_lr_scheduler_threshold=None, multi_step_lr_scheduler_milestones=None, exponential_lr_scheduler_gamma=None, use_pe=False, use_prune=False):
    assert lr_scheduler in ['reduce_on_plateau', 'exponential', 'multi_step',
                            None], 'LR scheduler can only be [reduce_on_plateau, exponential, multi_step]!'

    if gc.log_path != None:
        checkpoint_dir = os.path.join(gc.log_path, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

    if gc.dataset == "mosi":
        ds = MosiDataset
    elif gc.dataset == "mosi_unaligned":
        ds = MosiDatasetUnaligned
    elif gc.dataset == "mosei":
        ds = MoseiDataset
    elif gc.dataset == "mosei_unaligned":
        ds = MoseiDatasetUnaligned
    elif gc.dataset == "iemocap_unaligned":
        ds = IemocapDatasetUnaligned
    elif gc.dataset == "iemocap":
        ds = IemocapDataset
    else:
        ds = MoseiDataset

    train_dataset = ds(gc.data_path, clas="train")
    test_dataset = ds(gc.data_path, clas="test")
    valid_dataset = ds(gc.data_path, clas="valid")

    train_loader, train_labels = get_loader(train_dataset), train_dataset[:][-1]
    valid_loader, valid_labels = get_loader(valid_dataset), valid_dataset[:][-1]
    test_loader, test_labels = get_loader(test_dataset), test_dataset[:][-1]

    model = HeteroGNN(hidden_channels=64, out_channels=1, num_layers=6)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=gc.config['global_lr'],
        weight_decay=gc.config['weight_decay'],
        betas=(gc.config['beta1'], gc.config['beta2']),
        eps=gc.config['eps']
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=.002)

    from sklearn.metrics import accuracy_score

    best_mae = 1
    best_test_acc = 0
    best_valid_acc = 0
    for epoch in range(1, gc.config['epoch_num']):
        loss, train_acc, mae = train(train_loader, model, optimizer)
        valid_loss, valid_acc, valid_mae = test(valid_loader, model, scheduler, valid=True)
        test_loss, test_acc, test_mae = test(test_loader, model, scheduler, valid=False)
        if mae < best_mae:
            best_mae = mae
            best_test_acc = test_acc
            best_valid_acc = valid_acc

        print('Model parameters:', count_params(model))
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, ' f'Valid: {valid_acc:.4f}, Test: {test_acc:.4f}')

    print(f'\nBest test acc: {best_test_acc:.4f},\nBest valid acc: {best_valid_acc:.4f}')



def get_arguments():
    parser = standard_grid.ArgParser()
    # Network parameters
    parser.register_parameter("--optimizer", str, 'adam')
    parser.register_parameter("--exclude_vision", bool, False)
    parser.register_parameter("--exclude_audio", bool, False)
    parser.register_parameter("--exclude_text", bool, False)
    parser.register_parameter("--batch_size", int, 2)
    parser.register_parameter("--epoch_num", int, 50)
    parser.register_parameter("--cuda", int, 0)
    parser.register_parameter("--global_lr", float, 1e-4)
    parser.register_parameter("--gru_lr", float, 1e-4)
    parser.register_parameter("--beta1", float, 0.9)
    parser.register_parameter("--beta2", float, 0.999)
    parser.register_parameter("--eps", float, 1e-8)
    parser.register_parameter('--weight_decay', float, 1e-2)
    parser.register_parameter('--momentum', float, 0.9)
    parser.register_parameter("--gnn_dropout", float, 0.1)
    parser.register_parameter("--num_modality", int, 3)
    parser.register_parameter("--num_frames", int, 50)
    parser.register_parameter("--temporal_connectivity_order", int, 5)
    parser.register_parameter("--num_vision_aggr", int, 1)
    parser.register_parameter("--num_text_aggr", int, 1)
    parser.register_parameter("--num_audio_aggr", int, 1)
    parser.register_parameter("--text_dim", int, 300)
    parser.register_parameter("--audio_dim", int, 5)
    parser.register_parameter("--vision_dim", int, 20)
    parser.register_parameter("--graph_conv_in_dim", int, 512)
    parser.register_parameter("--graph_conv_out_dim", int, 512)
    parser.register_parameter("--use_same_graph_in_out_dim", int, 0)
    parser.register_parameter("--gat_conv_num_heads", int, 4)
    parser.register_parameter("--useGNN", int, 1, "whether to use GNN, 1 is yes, 0 is no.")
    parser.register_parameter("--average_mha", int, 0, "whether to average MHA for GAT, 1 is yes, 0 is no.")
    parser.register_parameter("--num_gat_layers", int, 1, "number of GAT layers")
    parser.register_parameter("--lr_scheduler", str, None,
                              "LR scheduler to use: 'reduce_on_plateau', 'exponential', 'multi_step'")
    parser.register_parameter("--reduce_on_plateau_lr_scheduler_patience", int, None, 10)
    parser.register_parameter("--reduce_on_plateau_lr_scheduler_threshold", float, None, 0.002)
    parser.register_parameter("--multi_step_lr_scheduler_milestones", str, None, "multi step lr schedule cutoff points")
    parser.register_parameter("--exponential_lr_scheduler_gamma", float, None, "exponential lr schedule decay factor")
    parser.register_parameter("--use_pe", int, 0, "whether to use positional embedding, 1 is yes, 0 is no")
    parser.register_parameter("--use_prune", int, 0, "whether to use pruning, 1 is yes, 0 is no")
    parser.register_parameter("--prune_keep_p", float, 0.75, "the percentage of nodes to keep")
    parser.register_parameter("--use_ffn", int, 0, "whether to use ffn in place of the linear projection layer before "
                                                   "entering graph")
    parser.register_parameter("--graph_activation", str, None, "type of activation to use between GAT layers")
    parser.register_parameter("--loss_type", str, "mse", "Loss to use to train model")
    parser.register_parameter("--remove_isolated", int, 0, "whether to remove isolated nodes")
    parser.register_parameter("--use_conv1d", int, 0,
                              "whether to use conv1d for vision (maybe in future for text/audio)")

    parser.register_parameter("--use_residual", int, 0, "whether to use residual connection between GAT layers")
    parser.register_parameter("--use_loss_norm", int, 0, "whether to use loss normalization")
    parser.register_parameter("--use_all_to_all", int, 0, "whether to use all-to-all connection in uni-modal nodes")
    parser.register_parameter("--checkpoints", str, '9,12,15,20', "checkpoints to peek test/val performance")
    parser.register_parameter("--use_iemocap_inverse_sample_count_ce_loss", int, 0, "whether to use iemocap_inverse_sample_count_ce_loss")

    # masking out certain modalities
    parser.register_parameter("--zero_out_video", int, 0, "whether to zero out video modality")
    parser.register_parameter("--zero_out_text",  int, 0, "whether to zero out text modality")
    parser.register_parameter("--zero_out_audio", int, 0, "whether to zero out audio modality")

    # Other settings, these are likely to be fixed all the time
    parser.register_parameter("--task", str, 'mosei', "task you are doing. Choose from mosi or mosei")
    parser.register_parameter("--dataroot", str, '/home/username/MTGAT/dataset/cmu_mosei', "path to the dataset")
    parser.register_parameter("--log_dir", str, None, 'log path for models')
    parser.register_parameter("--eval", bool, False, "whether this is a evaluation run")
    parser.register_parameter("--resume_pt", str, None, "the model pt to resume from")

    parser.register_parameter("--single_gpu", bool, True)
    parser.register_parameter("--load_model", bool, False)
    parser.register_parameter("--save_grad", bool, False)
    parser.register_parameter("--dataset", str, "mosi")
    parser.register_parameter("--data_path", str, "/workspace/dataset/")
    parser.register_parameter("--log_path", str, None)
    parser.register_parameter("--padding_len", int, -1)
    parser.register_parameter("--include_zero", bool, True)
    # for ablation
    # TODO：1. add flag to collapse past/present/future edge types into one type
    # TODO: 2. add flag gto collapse all 27 edge types into one type
    # TODO: 3. Add a flag to select random drop vs topk pruning
    parser.register_parameter('--time_aware_edges', int, 1, 'whether to use past current future to define edges')
    parser.register_parameter('--type_aware_edges', int, 1, 'whether to use node types to define edges')
    parser.register_parameter('--prune_type', str, 'topk', 'either to use topk or random to prune')

    parser.register_parameter('--dummy', int, 0, 'a dummy value for multiple-run of a same code')
    parser.register_parameter('--seed', int, 0, 'random seed')
    parser.register_parameter('--return_layer_outputs', int, 0, 'random seed')

    # for saving model
    parser.register_parameter('--save_best_model', int, 0, 'whether to save the best model')
    return parser.compile_argparse()


def get_arguments_argparse():
    import argparse
    parser = argparse.ArgumentParser()
    # Network parameters
    # parser.add_argument('-f', None)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--exclude_vision", type=bool, default=False)
    parser.add_argument("--exclude_audio", type=bool, default=False)
    parser.add_argument("--exclude_text", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--global_lr", type=float, default=1e-4)
    parser.add_argument("--gru_lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--gnn_dropout", type=float, default=0.1)
    parser.add_argument("--num_modality", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--temporal_connectivity_order", type=int, default=5)
    parser.add_argument("--num_vision_aggr", type=int, default=1)
    parser.add_argument("--num_text_aggr", type=int, default=1)
    parser.add_argument("--num_audio_aggr", type=int, default=1)
    parser.add_argument("--text_dim", type=int, default=300)
    parser.add_argument("--audio_dim", type=int, default=5)
    parser.add_argument("--vision_dim", type=int, default=20)
    parser.add_argument("--graph_conv_in_dim", type=int, default=512)
    parser.add_argument("--graph_conv_out_dim", type=int, default=512)
    parser.add_argument("--use_same_graph_in_out_dim", type=int, default=0)
    parser.add_argument("--gat_conv_num_heads", type=int, default=4)
    parser.add_argument("--useGNN", type=int, default=1, help="whether to use GNN, 1 is yes, 0 is no.")
    parser.add_argument("--average_mha", type=int, default=0, help="whether to average MHA for GAT, 1 is yes, 0 is no.")
    parser.add_argument("--num_gat_layers", type=int, default=1, help="number of GAT layers")
    parser.add_argument("--lr_scheduler", type=str, default=None,
                              help="LR scheduler to use: 'reduce_on_plateau', 'exponential', 'multi_step'")
    parser.add_argument("--reduce_on_plateau_lr_scheduler_patience", type=int, default=None, help=10)
    parser.add_argument("--reduce_on_plateau_lr_scheduler_threshold", type=float, default=None, help=0.002)
    parser.add_argument("--multi_step_lr_scheduler_milestones", type=str, default=None, help="multi step lr schedule cutoff points")
    parser.add_argument("--exponential_lr_scheduler_gamma", type=float, default=None, help="exponential lr schedule decay factor")
    parser.add_argument("--use_pe", type=int, default=0, help="whether to use positional embedding, 1 is yes, 0 is no")
    parser.add_argument("--use_prune", type=int, default=0, help="whether to use pruning, 1 is yes, 0 is no")
    parser.add_argument("--prune_keep_p", type=float, default=0.75, help="the percentage of nodes to keep")
    parser.add_argument("--use_ffn", type=int, default=0, help="whether to use ffn in place of the linear projection layer before "
                                                   "entering graph")
    parser.add_argument("--graph_activation", type=str, default=None, help="type of activation to use between GAT layers")
    parser.add_argument("--loss_type", type=str, default="mse", help="Loss to use to train model")
    parser.add_argument("--remove_isolated", type=int, default=0, help="whether to remove isolated nodes")
    parser.add_argument("--use_conv1d", type=int, default=0,
                              help="whether to use conv1d for vision (maybe in future for text/audio)")

    parser.add_argument("--use_loss_norm", type=int, default=0, help="whether to use loss normalization")
    parser.add_argument("--use_all_to_all", type=int, default=0, help="whether to use all-to-all connection in uni-modal nodes")
    parser.add_argument("--checkpoints", type=str, default='9,12,15,20', help="checkpoints to peek test/val performance")
    parser.add_argument("--use_iemocap_inverse_sample_count_ce_loss", type=int, default=0, help="whether to use iemocap_inverse_sample_count_ce_loss")

    # masking out certain modalities
    parser.add_argument("--zero_out_video", type=int, default=0, help="whether to zero out video modality")
    parser.add_argument("--zero_out_text",  type=int, default=0, help="whether to zero out text modality")
    parser.add_argument("--zero_out_audio", type=int, default=0, help="whether to zero out audio modality")

    # Other settings, these are likely to be fixed all the time
    parser.add_argument("--task", type=str, default='mosei', help="task you are doing. Choose from mosi or mosei")
    parser.add_argument("--dataroot", type=str, default='/home/username/MTGAT/dataset/cmu_mosei', help="path to the dataset")
    parser.add_argument("--log_dir", type=str, default=None, help='log path for models')
    parser.add_argument("--eval", type=bool, default=False, help="whether this is a evaluation run")
    parser.add_argument("--resume_pt", type=str, default=None, help="the model pt to resume from")

    parser.add_argument("--single_gpu", type=bool, default=True)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--save_grad", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="mosi")
    parser.add_argument("--data_path", type=str, default="/workspace/dataset/")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--padding_len", type=int, default=-1)
    parser.add_argument("--include_zero", type=bool, default=True)
    # for ablation
    # TODO：1. add flag to collapse past/present/future edge types into one type
    # TODO: 2. add flag gto collapse all 27 edge types into one type
    # TODO: 3. Add a flag to select random drop vs topk pruning
    parser.add_argument('--time_aware_edges', type=int, default=1, help='whether to use past current future to define edges')
    parser.add_argument('--type_aware_edges', type=int, default=1, help='whether to use node types to define edges')
    parser.add_argument('--prune_type', type=str, default='topk', help='either to use topk or random to prune')

    parser.add_argument('--dummy', type=int, default=0, help='a dummy value for multiple-run of a same code')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--return_layer_outputs', type=int, default=0, help='random seed')
    parser.add_argument('--save_best_model', type=int, default=0, help='whether to save model')
    parser.add_argument('--use_residual', type=int, default=0, help='whether to use residual connection')
    return parser.parse_args()

def assign_args_to_gc(args):
    # gc.config['optimizer'] = args.optimizer
    # gc.config['exclude_vision'] = args.exclude_vision
    # gc.config['exclude_audio'] = args.exclude_audio
    # gc.config['exclude_text'] = args.exclude_text
    gc.config['batch_size'] = args.batch_size
    gc.config['epoch_num'] = args.epoch_num
    gc.config['cuda'] = args.cuda
    gc.config['global_lr'] = args.global_lr
    gc.config['gru_lr'] = args.gru_lr
    gc.config['beta1'] = args.beta1
    gc.config['beta2'] = args.beta2
    gc.config['eps'] = args.eps
    gc.config['weight_decay'] = args.weight_decay
    gc.config['momentum'] = args.momentum
    gc.config['gnn_dropout'] = args.gnn_dropout
    gc.config['num_modality'] = args.num_modality
    gc.config['num_frames'] = args.num_frames
    gc.config['temporal_connectivity_order'] = args.temporal_connectivity_order
    gc.config['num_vision_aggr'] = args.num_vision_aggr
    gc.config['num_text_aggr'] = args.num_text_aggr
    gc.config['num_audio_aggr'] = args.num_audio_aggr
    gc.config['text_dim'] = args.text_dim
    gc.config['audio_dim'] = args.audio_dim
    gc.config['vision_dim'] = args.vision_dim
    gc.config['graph_conv_in_dim'] = args.graph_conv_in_dim
    gc.config['num_gat_layers'] = args.num_gat_layers
    gc.config['use_prune'] = args.use_prune
    gc.config['use_pe'] = args.use_pe
    gc.config['use_same_graph_in_out_dim'] = args.use_same_graph_in_out_dim
    if args.use_same_graph_in_out_dim:
        gc.config['graph_conv_out_dim'] = args.graph_conv_in_dim
    else:
        gc.config['graph_conv_out_dim'] = args.graph_conv_out_dim
    gc.config['gat_conv_num_heads'] = args.gat_conv_num_heads
    gc.config['prune_keep_p'] = args.prune_keep_p
    gc.config['use_ffn'] = args.use_ffn
    gc.config['graph_activation'] = args.graph_activation
    gc.config['loss_type'] = args.loss_type
    gc.config['remove_isolated'] = args.remove_isolated
    gc.config['use_conv1d'] = args.use_conv1d
    gc.config['use_loss_norm'] = args.use_loss_norm
    gc.config['use_all_to_all'] = args.use_all_to_all
    gc.config['checkpoints'] = [int(ckp) for ckp in args.checkpoints.split(',')]
    gc.config['use_iemocap_inverse_sample_count_ce_loss'] = args.use_iemocap_inverse_sample_count_ce_loss

    gc.config['zero_out_video'] = args.zero_out_video
    gc.config['zero_out_text'] = args.zero_out_text
    gc.config['zero_out_audio'] = args.zero_out_audio

    gc.config['time_aware_edges'] = args.time_aware_edges
    gc.config['type_aware_edges'] = args.type_aware_edges
    gc.config['prune_type'] = args.prune_type
    gc.config['seed'] = args.seed
    gc.config['return_layer_outputs'] = args.return_layer_outputs
    gc.config['save_best_model'] = args.save_best_model
    gc.config['use_residual'] = args.use_residual


if __name__ == "__main__":

    args = get_arguments()
    assert args.dataroot is not None, "You havn't provided the dataset path! Use the default one."
    gc.data_path = args.dataroot
    args.optimizer, args.task = args.optimizer.lower(), args.task.lower()
    assert args.task in ['mosi', 'mosei', 'mosi_unaligned', 'mosei_unaligned', 'iemocap', 'iemocap_unaligned'], "Unsupported task. Should be either mosi or mosei"
    gc.dataset = args.task

    assign_args_to_gc(args)
    if not args.eval:
        if args.log_dir is not None:
            now = datetime.now()
            now = now.strftime("%m-%d-%Y_T%H-%M-%S")
            gc.log_path = os.path.join(args.log_dir, now)
            if not os.path.exists(gc.log_path):
                os.makedirs(gc.log_path, exist_ok=True)
                log_file = os.path.join(gc.log_path, 'print.log')
                logging.basicConfig(level=logging.INFO)
                # logging.getLogger().addHandler(logging.FileHandler(log_file))
                # logging.getLogger().addHandler(logging.StreamHandler())

            # snapshot code to a zip file
            # util.snapshot_code_to_zip(code_path=pathlib.Path(__file__).parent.absolute(),
            #                           snapshot_zip_output_dir=gc.log_path,
            #                           snapshot_zip_output_file_name=f'code_snapshot_{now}.zip')

        start_time = time.time()
        logging.info('Start time: ' + time.strftime("%H:%M:%S", time.gmtime(start_time)))
        # torch.manual_seed(gc.config['seed'])
        util.set_seed(gc.config['seed'])
        best_results = train_model(args.optimizer,
                                   use_gnn=args.useGNN,
                                   average_mha=args.average_mha,
                                   num_gat_layers=args.num_gat_layers,
                                   lr_scheduler=args.lr_scheduler,
                                   reduce_on_plateau_lr_scheduler_patience=args.reduce_on_plateau_lr_scheduler_patience,
                                   reduce_on_plateau_lr_scheduler_threshold=args.reduce_on_plateau_lr_scheduler_threshold,
                                   multi_step_lr_scheduler_milestones=args.multi_step_lr_scheduler_milestones,
                                   exponential_lr_scheduler_gamma=args.exponential_lr_scheduler_gamma,
                                   use_pe=args.use_pe,
                                   use_prune=args.use_prune)
        elapsed_time = time.time() - start_time
        logging.info('Total time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        # log the results to grid search paths
        out_dir = "output/"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        res_f = open(os.path.join(out_dir, "best.txt"), "w")
        res_f.write(json.dumps(best_results))
    else:
        assert args.resume_pt is not None
        log_path = os.path.dirname(os.path.dirname(args.resume_pt))
        log_file = os.path.join(log_path, 'eval.log')
        logging.basicConfig(level=logging.INFO)
        logging.getLogger().addHandler(logging.FileHandler(log_file))
        # logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Start evaluation Using model from {}".format(args.resume_pt))
        start_time = time.time()
        eval_model(args.resume_pt)
        logging.info("Total evaluation time: {}".format(
            time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        )
