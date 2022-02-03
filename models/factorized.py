from .common import *

def s(x):
    save_pk('hi.pk', x)

def lv(x):
    return (load_pk('hi.pk')==x).all()

qa_conns = [
    # ('text', 'text_q', 'q'),
    # ('audio', 'audio_q', 'q'), 
    # ('video', 'video_q', 'q'), 
    
    # ('text', 'text_a', 'a'), 
    # ('audio', 'audio_a', 'a'), 
    # ('video', 'video_a', 'a'),

    # # flipped above
    # ('q', 'q_text','text'),
    # ('q', 'q_audio','audio'), 
    # ('q', 'q_video','video'), 
    
    # ('a', 'a_text','text'), 
    # ('a', 'a_audio','audio'), 
    # ('a', 'a_video','video'),

    ('q', 'q_a', 'a'),
    ('a', 'a_q', 'q'),

    # a to i
    # ('a', 'ai', 'a'),

    # self conns
    ('a', 'a_a', 'a'),
    ('q', 'q_q', 'q'),
]

z_conns = [
    ('text', 'text_z', 'z'),
    ('z', 'z_text', 'text'),
    ('audio', 'audio_z', 'z'),
    ('z', 'z_audio', 'audio'),
    ('video', 'video_z', 'z'),
    ('z', 'z_video', 'video'),
    ('z', 'z_z', 'z'),
    ('q', 'q_z', 'z'),
    ('z', 'z_q', 'q'),
    ('a', 'a_z', 'z'),
    ('z', 'z_a', 'a'),
]

non_mod_nodes = ['q', 'a', 'a_idx', 'i_idx', 'z']


def get_loader_solograph_chunk(ds, vad_intervals, dsname, gc):
    # total_data = load_pk(dsname)
    # total_data = None
    # if total_data is None:
    print('Chunk graphs')
    print(f'Regenerating graphs for {dsname}')
    keys, intervals = ds[-2:]

    if 'social' in gc['dataset']:
        q,a,inc=[torch.from_numpy(data[:]) for data in ds[0]]
        
        q = q.reshape(-1, q.shape[1]*q.shape[2], q.shape[-2], q.shape[-1]) # [888, 6, 12, 1, 25, 768] -> [888, 72, 25, 768]
        a = a.reshape(-1, a.shape[1]*a.shape[2], a.shape[-2], a.shape[-1]) # [888, 6, 12, 1, 25, 768] -> [888, 72, 25, 768]
        inc = inc.reshape(-1, inc.shape[1]*inc.shape[2], inc.shape[-2], inc.shape[-1]) # [888, 6, 12, 1, 25, 768] -> [888, 72, 25, 768]

        facet=torch.from_numpy(ds[1][:,:,:].transpose(1,0,2))
        words=torch.from_numpy(ds[2][:,:,:].transpose(1,0,2))
        covarep=torch.from_numpy(ds[3][:,:,:].transpose(1,0,2))

        # trim to just word keys
        word_keys = load_pk(f'{dsname}_word_keys.pk')
        idxs = ar([keys.index(elt) for elt in word_keys])
        assert (np.sort(idxs)==idxs).all(), 'word_keys are sorted differently from chunk keys'
        q = q[idxs]
        a = a[idxs]
        inc = inc[idxs]
        facet = facet[idxs]
        words = words[idxs]
        covarep = covarep[idxs]
        keys = ar(keys)[idxs]
        intervals = ar(intervals)[idxs]

        gc['true_bs'] = gc['bs']*q.shape[1] # bs refers to number of videos processed at once, but each q-a-mods is a different graph

    total_data = []
    num_skipped = 0
    num_smaller_ints = 0
    new_keys = []
    for i in tqdm(range(words.shape[0])):
        key = keys[i]
        data = {
            'text': get_masked(words[i]),
            'audio': get_masked(covarep[i]),
            'video': get_masked(facet[i]),
        }

        if not ( (len(data['text']) == len(data['audio'])) and (len(data['text']) == len(data['video'])) ): # skip where missing
            num_skipped += 1
            continue

        if gc['zero_out_video']:
            data['video'][:]=0
        if gc['zero_out_audio']:
            data['audio'][:]=0
        if gc['zero_out_text']:
            data['text'][:]=0
        
        if sum([len(v) for v in data.values()]) == 0:
            continue
        
        # split up into multiple subgraphs using VAD info: each word that begins within the utterance boundaries is part of that utterance
        ints = intervals[i]
        if key not in vad_intervals:
            num_skipped += 1
            continue

        vad_ints = vad_intervals[key][:,:-1] # ignore speaker id for now
        datas = []
        offset = 0
        z_offset = 0
        
        if len(ints) < len(vad_ints): # special case where fewer intervals than utterance boundaries; use intervals - one "word" per utterance
            vad_ints = np.arange(ints.shape[0])
            smaller_ints = True
        else:
            smaller_ints = False

        if len(vad_ints)==1:
            vad_ints[0][:] = [ints.min(), ints.max()]

        max_idx = -1
        for interval in vad_ints:
            if smaller_ints:
                idx = ar([interval])
            else:
                idx = np.where((ints[:,0] >= interval[0]) & (ints[:,0] <= interval[1]))[0]
            if len(idx) == 0:
                continue
            else:
                idx = idx[idx>max_idx] # exclude idxs already allocated
                max_idx = idx.max()
        
            _data = {
                'text': data['text'][idx],
                'text_idx': torch.arange(len(idx)) + offset,

                'audio': data['audio'][idx],
                'audio_idx': torch.arange(len(idx)) + offset,

                'video': data['video'][idx],
                'video_idx': torch.arange(len(idx)) + offset,
                
                'z': data['text'][idx].mean(axis=0)[None,:].repeat(gc['num_agg_nodes'],1),
                'z_idx': torch.arange(gc['num_agg_nodes']) + z_offset,
            }

            offset += len(idx)
            z_offset += gc['num_agg_nodes']
            datas.append(_data)
        
        z_idxs = torch.cat([elt['z_idx'] for elt in datas])
        if len(datas) == 0:
            assert len(datas) > 0

        num_utts = len(vad_ints)

        # ## TEST
        # datas = [
        #     {
        #         'text': torch.zeros(3,10),
        #         'audio': torch.zeros(3,10),
        #         'video': torch.zeros(3,10),
        #         'text_idx': torch.arange(3),
        #         'audio_idx': torch.arange(3),
        #         'video_idx': torch.arange(3),
        #         'z': torch.zeros(2,10),
        #         'z_idx': torch.arange(2),
        #     },
        #     {
        #         'text': torch.zeros(2,10),
        #         'audio': torch.zeros(2,10),
        #         'video': torch.zeros(2,10),
        #         'text_idx': torch.arange(2)+3,
        #         'audio_idx': torch.arange(2)+3,
        #         'video_idx': torch.arange(2)+3,
        #         'z': torch.zeros(2,10),
        #         'z_idx': torch.arange(2)+2,
        #     },
        #     {
        #         'text': torch.zeros(2,10),
        #         'audio': torch.zeros(2,10),
        #         'video': torch.zeros(2,10),
        #         'text_idx': torch.arange(2)+5,
        #         'audio_idx': torch.arange(2)+5,
        #         'video_idx': torch.arange(2)+5,
        #         'z': torch.zeros(2,10),
        #         'z_idx': torch.arange(2)+4,
        #     }
        # ]

        for j,data in enumerate(datas): # for each speaker turn
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
            
            # get z idxs
            for mod in mods:
                ## connect z to nodes via different modality connections
                data[f'{mod}', f'{mod}_z', 'z'] = get_fc_edges(data[f'{mod}_idx'], data['z_idx'])
                # outgoing as well as incoming
                data['z', f'z_{mod}', f'{mod}'] = data[f'{mod}', f'{mod}_z', 'z'].flip(dims=[0])

                ## connect z to mod nodes via same connection
                # data[f'{mod}', f'mod_nodes_z', 'z'] = torch.cat([elt[None,:] for elt in torch.meshgrid(data['z_idx'], data[f'{mod}_idx'])]).reshape(2,-1)
                # # outgoing as well as incoming
                # data['z', f'z_mod_nodes', f'{mod}'] = data[f'{mod}', f'mod_nodes_z', 'z'].flip(dims=[0])

            datas[j] = data
        
        # aggregate graph with offsets
        data = {
            k: torch.cat([elt[k] for elt in datas if k in elt], dim=(1 if isinstance(k,tuple) else 0)) 
            for k in datas[0].keys()
        }

        # connect z nodes together
        data['z', 'z_z', 'z'] = get_fc_edges(z_idxs, z_idxs)

        for j in range(q.shape[1]): # for each of the 72 q-a pairs, make a new graph
            # Q-A connections
            _q = q[i,j][None,:,:] # [1,25,768]
            _a = a[i,j][None,:,:] # [1,25,768]
            _inc = inc[i,j][None,:,:] # [1,25,768]

            if np.random.random() < .5: # randomly flip if answer or incorrect is first; shouldn't matter in graph setup
            # if True:
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

            # # Q/A - MOD
            # for mod in mods:
            #     data['q', f'q_{mod}', mod] = torch.cat([torch.zeros(data[mod].shape[0])[None,:], torch.arange(data[mod].shape[0])[None,:]], dim=0).to(torch.long) # [ [0,0,...0], [0,1,...len(mod)]]
            #     data[mod, f'{mod}_q', 'q'] = torch.clone(data['q', f'q_{mod}', mod]).flip(dims=[0])
            
            #     data['a', f'a_{mod}', mod] = torch.cat([
            #         torch.cat([torch.zeros(data[mod].shape[0])[None,:], torch.arange(data[mod].shape[0])[None,:]], dim=0).to(torch.long),
            #         torch.cat([torch.ones(data[mod].shape[0])[None,:], torch.arange(data[mod].shape[0])[None,:]], dim=0).to(torch.long)
            #     ], dim=-1)
            #     data[mod, f'{mod}_a', 'a'] = torch.clone(data['a', f'a_{mod}', mod]).flip(dims=[0])

            # Q/A - Z
            # q-z
            data['q', 'q_z', 'z'] = get_fc_edges(torch.zeros(1,dtype=torch.int64), z_idxs)
            data['z', 'z_q', 'q'] = torch.clone(data['q', 'q_z', 'z']).flip(dims=[0])

            # a-z
            data['a', 'a_z', 'z'] = get_fc_edges(a_idx, z_idxs)
            data['z', 'z_a', 'a'] = torch.clone(data['a', 'a_z', 'z']).flip(dims=[0])

            # Q-A
            data['q', 'q_a', 'a'] = torch.Tensor([ [0,0], [0,1] ]).to(torch.long)
            data['a', 'a_q', 'q'] = torch.clone(data['q', 'q_a', 'a']).flip(dims=[0])

            # Q-Q
            data['q', 'q_q', 'q'] = torch.Tensor([ [0], [0]]).to(torch.long)
            
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
            total_data.append(hetero_data)
        
        if i == 12 and (gc['test'] or gc['graph_test']):
            break
        
        new_keys.append(key)
        
    save_pk(f'{dsname}_keys.pk', new_keys)
    print('Total: ', words.shape[0])
    print('Num skipped: ', num_skipped)
    print('Num smaller: ', num_smaller_ints)
    # save_pk(dsname, total_data)

    # testing
    # loader = DataLoader(total_data, batch_size=2, shuffle=False)
    # if 'train' in dsname:
    #     batch = next(iter(loader))
    #     assert torch.all(batch['q', 'q_text', 'text']['edge_index'] == torch.Tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]]).to(torch.long)).item()
    #     assert torch.all(batch['text', 'text_q', 'q']['edge_index'] == torch.Tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]).to(torch.long)).item()

    # true_bs = gc['true_bs']
    # assert true_bs == gc['bs']*72

    # m=0
    # td = total_data[m*true_bs : (m+1)*true_bs]
    # qsize = 72
    # bs = gc['bs']
    # td_qs = [ # 3,72,25,768
    #     torch.cat([ elt['q']['x'] for elt in td[qnum*qsize:(qnum+1)*qsize] ])
    #     for qnum in range(bs)
    # ]
    # s(td_qs)

    ## make sure total_data matches output of batch process
    # loader = DataLoader(total_data, batch_size=true_bs, shuffle=False)
    # for m,batch in enumerate(tqdm(loader)):
    #     if batch['q']['x'].shape[0] != gc['true_bs']:
    #         print('Skipping last batch')
    #         break
        
    #     # Q
    #     td = total_data[m*true_bs : (m+1)*true_bs]
        
    #     qsize = 72
    #     bs = gc['bs']
    #     td_qs = [ # 3,72,25,768
    #         torch.cat([ elt['q']['x'] for elt in td[qnum*qsize:(qnum+1)*qsize] ])
    #         for qnum in range(bs)
    #     ]
    #     batch_qs = batch['q']['x'] # 3*72,qrep
    #     assert (batch_qs == torch.cat(td_qs)).all()

    #     # assert ar([(batch['q']['x'][k]==td[k]['q']['x']).all() for k in range(true_bs)]).all()
        
    #     # A
    #     td_as =  torch.cat( [td[k]['a']['x'] for k in range(true_bs)] )
    #     batch_as = batch['a']['x']
    #     assert (batch_as==td_as).all()

    loader = DataLoader(total_data, batch_size=gc['true_bs'], shuffle=True)
    return loader,gc

def get_fc_edges_window(idx1, idx2, window):
    # window = 2 # if idxs differ by more than this, don't include them
    # e.g. idx1 = idx2 = torch.arange(5), window = 2
    # tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
    # [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4]])
    arr = torch.cat([elt[None,:] for elt in torch.meshgrid(idx1, idx2)]).reshape(2,-1)
    valid_idxs = torch.where(torch.abs(arr[1,:]-arr[0,:]) <= window)[0]
    arr = arr[:,valid_idxs]
    return arr
    
def get_fc_edges_pastfut_window(idx1, idx2, window, idx1_earlier):
    arr = get_fc_edges_window(idx1,idx2,window)
    if idx1_earlier:
        valid_idxs = torch.where(arr[0,:]<arr[1,:])[0]
    else:
        valid_idxs = torch.where(arr[0,:]>arr[1,:])[0]
    return arr[:,valid_idxs]
    
def get_loader_solograph_word(ds, vad_intervals, dsname, gc):
    # total_data = load_pk(dsname)
    # total_data = None
    # if total_data is None:
    print(f'Regenerating graphs for {dsname}')
    keys, intervals = ds[-2:]
    new_keys = []

    save_pk(f'{dsname}_word_keys.pk', keys)

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
    num_skipped = 0
    num_smaller_ints = 0
    incorrectly_sorted = []
    for i in tqdm(range(words.shape[0])):
        # if i == 719:
            # hi=2
        # else:
            # continue
        key = keys[i]
        data = {
            'text': get_masked(words[i]),
            'audio': get_masked(covarep[i]),
            'video': get_masked(facet[i]),
        }

        if not ( (len(data['text']) == len(data['audio'])) and (len(data['text']) == len(data['video'])) ): # skip where missing
            num_skipped += 1
            continue

        if gc['zero_out_video']:
            data['video'][:]=0
        if gc['zero_out_audio']:
            data['audio'][:]=0
        if gc['zero_out_text']:
            data['text'][:]=0
        
        if sum([len(v) for v in data.values()]) == 0:
            num_skipped += 1
            continue
        
        # split up into multiple subgraphs using VAD info: each word that begins within the utterance boundaries is part of that utterance
        
        # assert intervals[i].shape[0]==data['text'].shape[0]

        word_ints = npr(intervals[i],2) # trust intervals for word timings passed through pipeline b/c of contraction splitting in feature extraction
        word_ints = word_ints[:gc['seq_len']] # trim to seqlen

        # TODO: temporary solution; ensure that idx is contiguous and increasing in start time - no words should be skipped; have to sort b/c of below problem coming out of MFA
        sorted_idxs = np.argsort(word_ints, axis=0)[:,0]
        if not np.all(sorted_idxs==np.arange(len(word_ints))): # if not consecutive
            incorrectly_sorted.append(key)
            word_ints = word_ints[sorted_idxs]
            data['text'] = data['text'][sorted_idxs]
        
        '''
        problem with speaker timings coming out of MFA requiring that we sort this array: 
            a = load_json('mfa_out.json')
            a['GzPIbX1pzDg']['intervals'][25:40]
            [[[9.657, 9.857], 'forward'],
            [[12.543, 13.433], 'this'],
            [[13.433, 14.863], 'election'],
            [[12.452, 12.512], 'go'],
            [[12.512, 12.672], 'ahead'],
            [[12.672, 16.452], 'mr'],
            [[16.452, 16.732], 'trump'],
            [[14.948, 14.978], 'i'],
            [[14.978, 15.248], 'have'],
            [[15.248, 15.648], 'heard'],
            [[15.648, 17.508], 'ted'],
            [[17.788, 17.918], 'say'],
            [[17.918, 18.668], 'that'],
            [[16.817, 16.927], 'over'],
            [[16.927, 17.527], 'and']]
        '''
        vad_ints = vad_intervals[key] # trust original all_utterances from process_MFA (MFA unified with diarization) to give us utterance boundaries

        if gc['graph_test']:
            # https://bit.ly/3tRt8z7
            data['text'] = tens(np.tile(np.arange(5), (6,1)))
            data['audio'] = tens(np.tile(np.arange(6)+5, (6,1)))
            data['video'] = tens(np.tile(np.arange(7)+5, (6,1)))

            word_ints = tens(np.sort(np.concatenate([np.arange(7), np.arange(7)[1:-1]])).reshape(-1,2))
            vad_ints = [ {'boundaries': [0,3.5]}, {'boundaries': [4.1,6]} ]

        # preprocess vad_ints so it's in correct order and has no duplicates (problem with MFA output)
        vad_ints = sorted(list(set([tuple(elt['boundaries']) for elt in vad_intervals[key]])), key=lambda x: x[0])

        offset = 0 # word offset as we split up into utterances
        z_offset = 0 # utterance offset, equal to utt_idx * gc['num_agg_nodes']
        datas = [] # to hold subgraph info for processing in next step

        # edge case: remove any utterances that do not have words in them
        idxs_to_keep = []
        vad_ints = ar(lzip(*vad_ints)).transpose()
        for utt_idx in range(len(vad_ints)):
            start,end = vad_ints[utt_idx]
            idx = np.where( ( (word_ints[:,0] >= start) & (word_ints[:,0] <= end) ) | ( (word_ints[:,1] >= start) &  (word_ints[:,1] <= end) ) )[0]
            if len(idx) > 0:
                idxs_to_keep.append(utt_idx)
        
        vad_ints = vad_ints[idxs_to_keep]

        for utt_idx in range(len(vad_ints)):
            start,end = vad_ints[utt_idx]
            
            # idxs of words in word_ints that correspond to this utterance
            idx = np.where( ( (word_ints[:,0] >= start) & (word_ints[:,0] <= end) ) | ( (word_ints[:,1] >= start) &  (word_ints[:,1] <= end) ) )[0]

            # assert that idx is contiguous, no words skipped.  Because of above sorting, this should be the case.
            '''
            this fails b/c of weird case below
            word_ints[110:118]
            array([[44.58, 44.83],
                [44.83, 48.63],
                [45.2 , 45.74],
                [45.74, 45.82],
                [45.82, 45.85],
                [45.85, 47.16],
                [47.18, 47.54],
                [47.54, 47.75]])
            start, end
            (47.435, 58.6)
            VPTjROKYhlU
            '''
            # assert np.all(idx == ( np.arange(len(idx)) + idx.min())) 
            assert (np.sort(idx) == idx).all() # a weaker condition: assert increasing
            
            _data = {
                'text': data['text'][idx],
                'text_idx': torch.arange(len(idx)) + offset,

                'audio': data['audio'][idx],
                'audio_idx': torch.arange(len(idx)) + offset,

                'video': data['video'][idx],
                'video_idx': torch.arange(len(idx)) + offset,
                
                'z': data['text'][idx].mean(axis=0)[None,:].repeat(gc['num_agg_nodes'],1),
                'z_idx': torch.arange(gc['num_agg_nodes']) + z_offset,
            }

            assert _data['text'].shape[0] == _data['audio'].shape[0] == _data['video'].shape[0] # assuming aligned data

            offset += len(idx)
            z_offset += gc['num_agg_nodes']

            if _data['text'].shape[0] > 0:
                datas.append(_data)
        
        z_idxs = torch.cat([elt['z_idx'] for elt in datas])

        # ## TEST
        # datas = [
        #     {
        #         'text': torch.zeros(3,10),
        #         'audio': torch.zeros(3,10),
        #         'video': torch.zeros(3,10),
        #         'text_idx': torch.arange(3),
        #         'audio_idx': torch.arange(3),
        #         'video_idx': torch.arange(3),
        #         'z': torch.zeros(2,10),
        #         'z_idx': torch.arange(2),
        #     },
        #     {
        #         'text': torch.zeros(2,10),
        #         'audio': torch.zeros(2,10),
        #         'video': torch.zeros(2,10),
        #         'text_idx': torch.arange(2)+3,
        #         'audio_idx': torch.arange(2)+3,
        #         'video_idx': torch.arange(2)+3,
        #         'z': torch.zeros(2,10),
        #         'z_idx': torch.arange(2)+2,
        #     },
        #     {
        #         'text': torch.zeros(2,10),
        #         'audio': torch.zeros(2,10),
        #         'video': torch.zeros(2,10),
        #         'text_idx': torch.arange(2)+5,
        #         'audio_idx': torch.arange(2)+5,
        #         'video_idx': torch.arange(2)+5,
        #         'z': torch.zeros(2,10),
        #         'z_idx': torch.arange(2)+4,
        #     }
        # ]

        for j,data in enumerate(datas):
            for mod in mods:
                # ret = build_time_aware_dynamic_graph_uni_modal(data[f'{mod}_idx'],[], [], 0, all_to_all=gc['use_all_to_all'], time_aware=True, type_aware=True)
                # 
                # data[mod, 'pres', mod] = get_fc_edges_window(data[f'{mod}_idx'], data[f'{mod}_idx'], window=gc['align_pres_window'])
                # for mod2 in [modx for modx in mods if modx != mod]: # other modalities
                for mod2 in mods:
                    data[mod, 'pres', mod2] = get_fc_edges_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pres_window'])
                    data[mod2, 'pres', mod] = get_fc_edges_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pres_window'])

                    data[mod, 'past', mod2] = get_fc_edges_pastfut_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pastfut_window'], idx1_earlier=False)
                    data[mod, 'fut', mod2] = get_fc_edges_pastfut_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pastfut_window'], idx1_earlier=True)

            # get z idxs
            for mod in mods:
                ## connect z to nodes via different modality connections
                data[f'{mod}', f'{mod}_z', 'z'] = get_fc_edges(data[f'{mod}_idx'], data['z_idx'])
                # outgoing as well as incoming
                data['z', f'z_{mod}', f'{mod}'] = data[f'{mod}', f'{mod}_z', 'z'].flip(dims=[0])

                ## connect z to mod nodes via same connection
                # data[f'{mod}', f'mod_nodes_z', 'z'] = torch.cat([elt[None,:] for elt in torch.meshgrid(data['z_idx'], data[f'{mod}_idx'])]).reshape(2,-1)
                # # outgoing as well as incoming
                # data['z', f'z_mod_nodes', f'{mod}'] = data[f'{mod}', f'mod_nodes_z', 'z'].flip(dims=[0])

            datas[j] = data
        
        # aggregate graph with offsets
        data = {
            k: torch.cat([elt[k] for elt in datas if k in elt], dim=(1 if isinstance(k,tuple) else 0)) 
            for k in datas[0].keys()
        }

        # connect z nodes together
        data['z', 'z_z', 'z'] = get_fc_edges(z_idxs, z_idxs)

        if gc['graph_test']:
            assert gc['align_pres_window'] == 1 and gc['align_pastfut_window'] == 10
            mod_pres = pairs_to_arr([(0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,3), (4,4), (4,5), (5,4), (5,5)])
            mod_fut_mod = pairs_to_arr([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (4,5)]) # TODO: this should be the case, but right now we have a window size 1
            mod_past_mod = pairs_to_arr([(1,0), (2,0), (3,0), (2,1), (3,1), (3,2), (5,4)]) # TODO: this should be the case, but right now we have a window size 1
            mod_z = pairs_to_arr([(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1)])
            z_mod = pairs_to_arr([(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])

            for mod in mods:
                arr1 = data[mod, f'{mod}_z', 'z']
                assert pointers_eq(arr1, mod_z)
                arr1 = data['z', f'z_{mod}', mod]
                assert pointers_eq(arr1, z_mod)

                for mod2 in mods:
                    for rel in ['past', 'pres', 'fut']:
                        arr1 = data[mod, rel, mod2]
                        if rel == 'pres':
                            arr2 = mod_pres
                        elif (rel == 'fut'):
                            arr2 = mod_fut_mod
                        elif (rel == 'past'):
                            arr2 = mod_past_mod
                        else:
                            assert False
                        
                        assert pointers_eq(arr1,arr2), pointers_diff(arr1, arr2)

        # q-z
        data['q', 'q_z', 'z'] = get_fc_edges(torch.zeros(1,dtype=torch.int64), z_idxs)
        data['z', 'z_q', 'q'] = torch.clone(data['q', 'q_z', 'z']).flip(dims=[0])

        # Q-AI
        data['q', 'q_a', 'a'] = torch.Tensor([ [0,0], [0,1] ]).to(torch.long)
        data['a', 'a_q', 'q'] = torch.clone(data['q', 'q_a', 'a']).flip(dims=[0])

        # Q-Q
        data['q', 'q_q', 'q'] = torch.Tensor([ [0], [0]]).to(torch.long)

        # A-A
        data['a', 'a_a', 'a'] = torch.Tensor([[0,0,1,1],[0,1,0,1]]).to(torch.long)
       
        for j in range(q.shape[1]): # for each of the 72 q-a pairs, make a new graph
            # Q-A connections
            _q = q[i,j][None,:,:] # [1,25,768]
            # assert _q.shape==(1,25,768)
            _a = a[i,j][None,:,:] # [1,25,768]
            _inc = inc[i,j][None,:,:] # [1,25,768]

            if np.random.random() < .5: # randomly flip if answer or incorrect is first; shouldn't matter in graph setup
            # if True:
                _a1 = _a
                _a2 = _inc
                a_idx = torch.Tensor([1,0]).to(torch.long)
                i_idx = torch.Tensor([0,1]).to(torch.long)

            else:
                _a1 = _inc
                _a2 = _a
                a_idx = torch.Tensor([0,1]).to(torch.long)
                i_idx = torch.Tensor([1,0]).to(torch.long)

            # a-z
            data['a', 'a_z', 'z'] = get_fc_edges(a_idx, z_idxs)
            data['z', 'z_a', 'a'] = torch.clone(data['a', 'a_z', 'z']).flip(dims=[0])

            _as = torch.cat([_a1, _a2], dim=0)

            hetero_data = {
                **{k: {'x': v} for k,v in data.items() if 'idx' not in k and not isinstance(k, tuple)}, # just get data on mods
                **{k: {'edge_index': v} for k,v in data.items() if isinstance(k, tuple) },
                'q': {'x': _q},
                'a': {'x': _as},
                'a_idx': {'x': a_idx},
                'i_idx': {'x': i_idx},
            }

            hetero_data = HeteroData(hetero_data) # different "sample" for each video-q-a graph
            total_data.append(hetero_data)
        
        if i == 12 and (gc['test'] or gc['graph_test']):
            break
            
        new_keys.append(key)
        
    save_pk(f'{dsname}_keys_word.pk', new_keys)
    print('Total: ', words.shape[0])
    print('Num skipped: ', num_skipped)
    print('Num smaller: ', num_smaller_ints)
    print('Num incorrectly sorted: ', len(incorrectly_sorted))
    # save_pk(dsname, total_data)

    # testing
    # loader = DataLoader(total_data, batch_size=2, shuffle=False)
    # if 'train' in dsname:
    #     batch = next(iter(loader))
    #     assert torch.all(batch['q', 'q_text', 'text']['edge_index'] == torch.Tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1], [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]]).to(torch.long)).item()
    #     assert torch.all(batch['text', 'text_q', 'q']['edge_index'] == torch.Tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35], [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]).to(torch.long)).item()

    # true_bs = gc['true_bs']
    # assert true_bs == gc['bs']*72

    ## make sure total_data matches output of batch process
    # loader = DataLoader(total_data, batch_size=true_bs, shuffle=False)
    # for m,batch in enumerate(tqdm(loader)):
    #     if batch['q']['x'].shape[0] != gc['true_bs']:
    #         print('Skipping last batch')
    #         break
    #     # Q
    #     td = total_data[m*true_bs : (m+1)*true_bs]
        
    #     qsize = 72
    #     bs = gc['bs']
    #     td_qs = [ # 3,72,25,768
    #         torch.cat([ elt['q']['x'] for elt in td[qnum*qsize:(qnum+1)*qsize] ])
    #         for qnum in range(bs)
    #     ]
    #     batch_qs = batch['q']['x'] # 3*72,qrep
    #     assert (batch_qs == torch.cat(td_qs)).all()

    loader = DataLoader(total_data, batch_size=gc['true_bs'], shuffle=True)
    return loader,gc


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
                for conn_type in all_connections + qa_conns + z_conns
            }, aggr='mean')
            
            self.convs.append(conv)

        self.pes = {k: PositionalEncoding(gc['graph_conv_in_dim']) for k in mods}

    def forward(self, x_dict, edge_index_dict, batch_dict):
        mod_dict = {k: v for k,v in x_dict.items() if k not in ['z', 'q', 'a']}
        qaz_dict = {k: v for k,v in x_dict.items() if k in ['z', 'q', 'a']}
        mod_dict = {key: self.lin_dict[key](x) for key, x in mod_dict.items()}
        qaz_dict['z'] = self.lin_dict['text'](qaz_dict['z'])

        # apply pe
        for m, v in mod_dict.items(): # modality, tensor
            idxs = batch_dict[m]
            assert (idxs==(idxs.sort().values)).all()
            _, counts = torch.unique(idxs, return_counts=True)
            mod_dict[m] = self.pes[m](v, counts)

        # init z
        # x_dict['z'] = 
        
        x_dict = {
            **mod_dict,
            **qaz_dict,
        }
        
        for i,conv in enumerate(self.convs):
            if gc['BASELINE'] and i == 0:
                save_pk('x_dict.pk', x_dict)
                save_pk('edge_index_dict.pk', edge_index_dict)
                gc['x_dict_old'] = x_dict
                gc['edge_index_dict_old'] = edge_index_dict

            x_dict, edge_types = conv(x_dict, edge_index_dict, return_attention_weights_dict={elt: True for elt in all_connections+qa_conns + z_conns})

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

