from alex_utils import *
from transformers import pipeline, AutoTokenizer

def split_words_speaker_turns(mfa,diar):
    '''
    in:
    mfa: (from /work/awilf/mfa), of form
    "29vnZjb39u0": {
        "wav_idx": 0,
        "intervals": [
            [
                [
                    2.82,
                    3.06
                ],
                "that's"
            ],
    diar: speaker diarization (from process_VAD.py), of form {k: [ [start,end,speaker_id] ]}
    it is "squashed" because it "squashes" all speaker turns into a single interval

    e.g.
    {'-daGjyKKNio': array([[ 0.   , 34.87 ,  0.   ],
            [36.31 , 45.19 ,  1.   ],
            [45.73 , 49.515,  0.   ],
            [49.515, 59.92 ,  1.   ]], dtype=float32),

    out:
    all_utterances, of form {k: [utterances]}, where [utterances] is an array of objects that look like this:
    {   
        'boundaries': array([30.525, 44.93 ], dtype=float32),
        'speaker_id': 0.0,
        'words': [   ((30.541, 30.951), "i've"),
                        ((31.151, 31.581), 'got'),
                        ((31.581, 31.931), 'turbines'),
                        ((31.931, 32.311), 'existing'),
                        ((32.311, 32.411), 'in'),
                        ((32.411, 32.521), 'our'),
                        ((32.521, 33.261), 'community'),
                        ((33.261, 33.401), 'now'),
                        ((33.401, 34.111), 'making'),
                        ((34.141, 34.441), 'people'),
                        ((34.49, 34.87), 'sick'),
                        ((34.87, 34.98), 'and'),
            ]
    },
    '''
    diar = {k.replace('_trimmed-out.wav', ''): v for k,v in diar.items()}

    all_utterances = {}

    for k in tqdm(lkeys(mfa)):
        # remove this video, b/c vad couldn't process it
        if k == 'F2mIH0vlI9c':
            continue

        utterances = []
        utt = {
            'boundaries': None,
            'speaker_id': None,
            'words': []
        }
        current_utt = 0
        words_dropped = 0
        for i, ((start,end),word) in enumerate(mfa[k]['intervals']):
            # print(start, end)
            idx_arr = ( (start >=diar[k][:,0]) & (start <= diar[k][:,1]) ) | ( (end >= diar[k][:,0]) &  (end <= diar[k][:,1]) )
            if not np.any(idx_arr): # word start / end is not in any interval
                words_dropped += 1
                continue
            
            idx = idx_arr.argmax()
            speaker_idx = diar[k][idx,2]

            if i == 0:
                current_utt = idx # in case there are intervals without any words in them to start

            if idx == current_utt:
                utt['boundaries'] = diar[k][idx][0:2]
                utt['speaker_id'] = diar[k][idx][2]
                utt['words'].append(((start,end),word))
            else:
                if utt['boundaries'] is not None:
                    utterances.append(utt)
                
                utt = [((start,end),word,speaker_idx)]

                utt = {
                    'boundaries': diar[k][idx][0:2],
                    'speaker_id': diar[k][idx][2],
                    'words': [ ((start,end),word) ],
                }
                current_utt = idx

        if utt['boundaries'] is not None:
            utterances.append(utt)
        all_utterances[k] = utterances
    return all_utterances

def get_bert_features(all_utterances, model_name='bert-base-uncased'):
    diar_failed = [] # videos where diarization failed
    to_ret = defaultdict(list)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feature_extraction = pipeline('feature-extraction', model=model_name, tokenizer=model_name)

    # Get new intervals by splitting up intervals for which there are multiple tokens (e.g. that's)
    for vid_id in tqdm(all_utterances.keys()):
        if len(all_utterances[vid_id]) == 0:
            diar_failed.append(vid_id)
            continue
        
        for utt_idx in range(len(all_utterances[vid_id])):
            utt = all_utterances[vid_id][utt_idx]
            intervals, words = lzip(*utt['words'])
            words = ' '.join(words)

            # get intervals for each word as processed by tokenizer (e.g. that's -> that ' s, so the interval should be split three ways)
            new_intervals,tokens = [],[]
            for interval,word in utt['words']:
                encoded = tokenizer.encode(word)
                assert len(encoded) >=3 
                encoded = encoded[1:-1] # remove start & end tokens
                
                if len(encoded) == 1: # normal word
                    new_intervals.append(ar([interval]))
                    tokens.append(encoded[0])
                
                else: # word contains punctuation, e.g. "i'm"
                    start,end = interval

                    space = np.linspace(start,end,len(encoded)+1)
                    space = np.concatenate([space,space[1:-1]]) # duplicate internal numbers for intervals
                    space.sort()
                    space = space.reshape(-1,2)

                    new_intervals.append(space)
                    tokens.extend(encoded)

            tokens = [101, *tokens, 102]
            full_encoded_seq = tokenizer.encode(words)
            assert subsets_equal(tokens, full_encoded_seq), 'Problem with the above code - intervals may be wrong.'
            new_intervals = np.concatenate(new_intervals)
            features = feature_extraction(words)

            to_ret[vid_id].append({
                'boundaries': utt['boundaries'],
                'speaker_id': utt['speaker_id'],
                'features':  ar(features),
                'intervals': ar(new_intervals),
                'words': words,
            })
    
    print('The following keys failed because of lack of diarization.  Check vad_intervals_squashed.pk', diar_failed)
    return to_ret

if __name__ == '__main__':
    mfa = load_json('mfa_out.json') # from /work/awilf/mfa
    diar = load_pk('data/vad_intervals_squashed.pk') # from process_VAD

    all_utterances = load_pknone('data/split_utterances.pk', split_words_speaker_turns, [mfa, diar])

    '''
    bert_features is of form:
    vid_id: [ 
    { # utterance 0
        'boundaries': (start,end), # within video
        'speaker_id': {0/1/2},
        'features': [1,n+2,768], # where n is the number of tokens in the utterance; [CLS] [*words tokens] [SEP]
        'intervals': [n,768],
        'words': str, original words in case you need them down the line.  Note: n may be different than len(words.split(' ')) because some words lead to multiple tokens (e.g. that's).
                We adjust the intervals to split up intervals evenly across these tokens in the above function.
    },
    {
        ...
    }
    ]
    '''

    bert_features = load_pknone('data/bert_features.pk', get_bert_features, [all_utterances])

    pk = {}
    for vid in bert_features:
        pk[vid] = {
            'features': np.concatenate([utt['features'].squeeze(0)[1:-1] for utt in bert_features[vid]]),
            'intervals': np.concatenate([utt['intervals'] for utt in bert_features[vid]])
        }
        assert pk[vid]['features'].shape[0] == pk[vid]['intervals'].shape[0]

    # pk_to_amir_csd(pk, 'bert_word_csd.csd')
    save_pk('data/bert_word_csd.pk', pk)
    
