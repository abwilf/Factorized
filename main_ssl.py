from torch.autograd import Variable

import traceback
import torch.utils.data as Data
from datetime import datetime
import json
import os
import sys
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import gc as g
from graph_model.iemocap_inverse_sample_count_ce_loss import IEMOCAPInverseSampleCountCELoss
from model import NetMTGATAverageUnalignedConcatMHA
import logging
import util
import pathlib
import random
from arg_defaults import defaults
from consts import GlobalConsts as gc
from os.path import dirname

from models.mylstm import MyLSTM
from models.social_iq import *
from models.alex_utils import *
from models.common import *
from models.graph_builder import construct_time_aware_dynamic_graph, build_time_aware_dynamic_graph_uni_modal, build_time_aware_dynamic_graph_cross_modal
from models.global_const import gc
from models.mosi import*
import sys; sys.path.append('/work/qianlim/mtag/Standard-Grid'); import standard_grid


import gc as g
from sklearn.metrics import accuracy_score

SEEDS = list(range(100))

ie_emos = ["Neutral", "Happy", "Sad", "Angry"]

def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_loader, dev_loader = None, None
def train_model_social():
    global train_loader, dev_loader, gc

    if gc['net'] == 'graphqa':
        from models.graphqa import get_loader_solograph, Solograph
        raise NotImplementedError
    
    elif gc['net'] == 'factorized':
        if gc['gran'] == 'chunk':
            from models.factorized import get_loader_solograph_chunk as get_loader_solograph
        else:
            from models.factorized import get_loader_solograph_word as get_loader_solograph
            
        from models.ssl import SolographContrastive
    else:
        raise NotImplementedError
        # assert gc['net'] == 'factorized', f'gc[net]needs to be factorized but is {gc["net"]}'

    model = SolographContrastive()
    if not os.path.exists(dirname(gc['model_path'])):
        os.makedirs(dirname(gc['model_path']))

    if train_loader is None: # cache train and dev loader so skip data loading in multiple iterations
        print('Building loaders for social')
        trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
        # These videos have some issues in training set
        bads=['f5NJQiY9AuY','aHBLOkfJSYI']
        folds=[trk,dek]
        for bad in bads:
            for fold in folds:
                try:
                    fold.remove(bad)
                except:
                    pass

        preloaded_train=process_data(trk, 'train', gc)
        preloaded_dev=process_data(dek, 'dev', gc)
        replace_inf(preloaded_train[3])
        replace_inf(preloaded_dev[3])

        # TODO: clean up global variables here
        # intervals_subpath = 'vad_intervals_squashed.pk' if gc['gran'] == 'chunk' else 'bert_features.pk'
        intervals = load_pk(join(gc['raw_data'], 'text', 'vad', 'full_bert_feats.pk'))
        if gc['net'] == 'graphqa':
            train_loader = get_loader_solograph(preloaded_train, 'social_train')
            dev_loader = get_loader_solograph(preloaded_dev, 'social_dev')
        else:
            train_loader, gc = get_loader_solograph(preloaded_train, intervals, 'social_train', gc)
            dev_loader, gc = get_loader_solograph(preloaded_dev, intervals, 'social_dev', gc)
        
        del preloaded_train
        del preloaded_dev

    #Initializing parameter optimizer
    model = model.to(gc['device'])
    #print(gc)
    params= list(model.q_lstm.parameters())+list(model.a_lstm.parameters())+list(model.projection_head.parameters())
    optimizer=optim.Adam(params,lr=gc['global_lr'])

    # graph optimizer
    graph_optimizer = torch.optim.AdamW(
        model.hetero_gnn_contrastive.parameters(),
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
        train_losses, train_accs = [],[]

        train_block = gc['train_block']
        for batch_i, batch in enumerate(tqdm(train_loader)):
            batch = batch.to(gc['device'])

            if batch_i == 0:
                with torch.no_grad():  # Initialize lazy modules.
                    loss = model(batch)
                    del loss
                    torch.cuda.empty_cache()
            
            cont = False
            for mod in mods:
                if not np.any([mod in elt for elt in batch.edge_index_dict.keys()]):
                    print(mod, 'dropped from train loader!')
                    cont = True
            if cont:
                continue
            
            for mod in mods:
                assert not batch[mod]['x'].isnan().any()
                assert not batch[mod]['batch'].isnan().any()
                assert not batch[mod]['ptr'].isnan().any()

            edges = lkeys(batch.__dict__['_edge_store_dict'])

            for (l,name,r) in edges: # check for out of bounds accesses
                assert batch[l]['x'].shape[0] >= batch[l,name,r]['edge_index'][0,:].max()
                assert batch[r]['x'].shape[0] >= batch[l,name,r]['edge_index'][1,:].max()
            
            loss = model(batch)

            optimizer.zero_grad()
            graph_optimizer.zero_grad()

            loss.backward()
            
            optimizer.step()
            graph_optimizer.step()

            train_losses.append(loss.cpu().detach().numpy())
            torch.cuda.empty_cache()

        train_loss = np.mean(train_losses)
        metrics['train_losses'].append(train_loss)
        print(f'Train loss:{train_loss:.4f}')
        torch.save(model.state_dict(), gc['model_path'])


    print('Model parameters:', count_params(model))
    return model

def get_arguments():
    parser = standard_grid.ArgParser()
    for arg in defaults:
        parser.register_parameter(*arg)

    args = parser.compile_argparse()

    global gc
    for arg, val in args.__dict__.items():
        gc[arg] = val
    
    gc['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    get_arguments() # updates gc

    assert gc['data_path'] is not None, "You havn't provided the dataset path! Use the default one."
    gc['data_path'] = join(gc['data_path'], gc['dataset'])
    gc['proc_data'] = join(gc['data_path'], 'processed')
    gc['csd_data'] = join(gc['data_path'], 'csd')
    gc['raw_data'] = join(gc['data_path'], 'raw')

    start_time = time.time()
    
    if 'mosi' in gc['dataset']:
        train_fn = train_model_mosi

    elif gc['social_baseline']:
        train_fn = train_social_baseline
        raise NotImplementedError
    else:
        train_fn = train_model_social
    
    all_results = []
    for trial in range(gc['trials']):
        print(f'\nTrial {trial}')
        util.set_seed(SEEDS[trial])
        model = train_fn()
