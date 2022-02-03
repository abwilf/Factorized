## TODO integrate this into main.py structure to get working on MOSI / MOSEI.
## NOTE: This code is NOT integrated with the rest of the project yet.  This is legacy code from a previous experiment. 

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



def train(train_loader, model, optimizer):
    total_loss, total_examples = 0,0
    y_trues = []
    y_preds = []
    model.train()
    if 'iemocap' in gc['dataset']:
        criterion = IEMOCAPInverseSampleCountCELoss()
        criterion.to(gc['device'])
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
        
        data = data.to(gc['device'])
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
        data = data.to(gc['device'])
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
    model = model.to(gc['device'])
    
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
