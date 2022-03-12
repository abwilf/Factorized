from transformers import Wav2Vec2Processor, HubertModel
import librosa
from math import ceil
import numpy as np
import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
# model = model.cuda()

def get_features(wav_file,optional_params=None):
    '''
    in: path to wav_file
    out: features,intervals
    features: [T,d] np array, where T is the number of timesteps and d is the dimensionality of each extracted vector
    intervals: [T,2] np array, containing start,end pairs for each slice.  e.g. [ [0,.1], [.1,.2], ...]

    T is determined by the sampling rate that you draw the features using.  I think covarep and other low-level 
    feature extractors as we use them are every .1 seconds. You might want to add options to change this depending on the models
    and what timesteps you find they perform best with.  You can control this with the optional_params flag; let me know, and we'll
    add the parameters passed in end-to-end from the top level arguments, then we can hyperparam search over them.
    '''
    
    # Load + resample + make single channel file
    audioInput, sr = librosa.load(wav_file, sr=16000, mono=True)    
    time = len(audioInput)/16000    
    T = ceil(time/(20/1000))
    
    # Produce timestamps
    indices1 = np.arange(0, T)
    indices2 = indices1+1
    indices1, indices2 = indices1*20/1000, indices2*20/1000
    timeStamps = np.stack([indices1, indices2], axis=1)
    timeStamps[-1][1] = time
    
    # Pad the end of the audio input with 0s
    desiredSize = 400+(T-1)*320
    audioInput = np.pad(audioInput, [0, desiredSize-len(audioInput)])
    
    # Calculate the embeddings (the hidden states in Hubert language)
    input_values = processor(audioInput, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
    hidden_states = model(input_values).last_hidden_state[0]
    
    return (hidden_states.detach().numpy(), timeStamps)


    # # input_values = processor(audioInput, return_tensors="pt", sampling_rate=16000).input_values.squeeze().cuda()  # Batch size 1
    # # hidden_states = model(input_values).last_hidden_state[0].detach().cpu()
    # # hidden_states = model(input_values).last_hidden_state[0]


    # # split into minibatches of size k
    # k = 10000
    # vals = input_values.split(input_values.shape[0]//k)
    # full_res = None
    # for val in tqdm(vals):
    #     hidden_states = model(val[None,:]).last_hidden_state[0].detach().cpu()
    #     if full_res is None:
    #         full_res = hidden_states
    #     else:
    #         full_res = torch.cat([full_res, hidden_states])

    # return (hidden_states.detach().cpu().numpy(), timeStamps)

def get_features_dir(wav_file_dir, save_path, optional_params=None):
    '''
    same thing as above, but done across a directory of wav_files.  The output is a dict of the form
    {
        wav_key: {
            'features': ...
            'intervals': ...
        }
    }
    it'd be nice if you implemented this instead of me, because you may be able to paralellize across multiple wavs
    within a single forward pass of the model on the GPU and this would be much faster than me running get_features n times
    '''
    pk = {}
    for wav_path in tqdm(glob(join(wav_file_dir,'*'))):
        wav_id = wav_path.split('/')[-1].rsplit('.',1)[0]
        feats, intervals = get_features(wav_path)
        pk[wav_id] = {
            'features': feats,
            'intervals': intervals,
        }
    
    save_pk(save_path, pk)
    
    return pk

get_features_dir('/work/awilf/MTAG/data/mosi/raw/audio', '/work/awilf/MTAG/hubert.pk')

