from .common import *


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

    loader = DataLoader(total_data, batch_size=gc['true_bs'], shuffle=True)
    return loader


class Solograph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_lstm = MyLSTM(768,gc['graph_conv_in_dim'])
        self.a_lstm = MyLSTM(768,gc['graph_conv_in_dim'])

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
            q_out, a_out, scene_rep = self.hetero_gnn(x_dict, batch.edge_index_dict, batch.batch_dict)

            a = a_out[torch.where(a_idx)[0]]
            inc = a_out[torch.where(i_idx)[0]]

            correct = self.judge(torch.cat((q_out, a, inc, scene_rep), 1))
            incorrect = self.judge(torch.cat((q_out, inc, a, scene_rep), 1))
        
        else:
            q_out, a_out = self.hetero_gnn(x_dict, batch.edge_index_dict, batch.batch_dict)

            a = a_out[torch.where(a_idx)[0]]
            inc = a_out[torch.where(i_idx)[0]]

            correct = self.judge(torch.cat((q_out, a, inc), 1))
            incorrect = self.judge(torch.cat((q_out, inc, a), 1))

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

    def forward(self, x_dict, edge_index_dict, batch_dict):
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

        # get mean scene rep
        if gc['scene_mean']:
            x = torch.cat([v for k,v in x_dict.items() if k not in non_mod_nodes], axis=0)
            batch_dicts = torch.cat([v for k,v in batch_dict.items() if k not in non_mod_nodes], axis=0)
            x = scatter_mean(x, batch_dicts, dim=0)
            scene_rep = x

            return x_dict['q'], x_dict['a'], scene_rep
        else:
            return x_dict['q'], x_dict['a']
