from voice_activity import get_VAD_intervals
from alex_utils import *
import os
from tqdm import tqdm

# all_intervals = {}
# out_dir = '/work/awilf/MTAG/VAD_output'
# wav_dir = '/work/awilf/social_iq_raw/acoustic/wav/'
# wav_paths = os.listdir(wav_dir)

# all_intervals = load_pk('vad_intervals.pk')
# for wav_path in tqdm(wav_paths):
#     if wav_path in all_intervals:
#         continue
#     try:
#         all_intervals[wav_path] = get_VAD_intervals(os.path.join(wav_dir,wav_path), out_dir)
#     except:
#         print(wav_path + ' could not be decoded into VAD intervals!')
#         all_intervals[wav_path] = None
    
#     save_pk('vad_intervals.pk', all_intervals)

## squash
import numpy as np
vad_intervals = load_pk('vad_intervals.pk')
vad_intervals = {k.split('_trimmed')[0]: v for k,v in vad_intervals.items()}
vad_intervals = {k: np.array([ [np.float32(elt2) for elt2 in elt.replace('speaker_', '').split(' ')] for elt in v]) for k,v in vad_intervals.items() if v is not None}

for k in vad_intervals.keys():
    arr = vad_intervals[k]

    # preprocess: squash vad_intervals so they correspond to intervals over which single speaker is speaking before change
    speaker = -1
    int_min, int_max = 0,0
    new_arr = []
    for i in range(arr.shape[0]):
        if speaker == -1:
            int_min,int_max,speaker = arr[0]

        elif arr[i,-1] == speaker: # update upper bound
            int_max = arr[i,1]

        else: # add prev, update both
            new_arr.append([int_min, int_max, speaker])
            int_min,int_max,speaker = arr[i]

    new_arr.append([int_min, int_max, speaker])
    new_arr = np.array(new_arr,dtype=np.float32)
    
    vad_intervals[k] = new_arr

save_pk('vad_intervals_squashed.pk', vad_intervals)
