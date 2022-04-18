import json
import os
import wget
import matplotlib.pyplot as plt
import numpy as np
#import librosa
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.vad_utils import plot
import nemo.collections.asr as nemo_asr

def get_VAD_intervals(wav_path, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    meta = {
        'audio_filepath': wav_path, 
        'offset': 0, 
        'duration':None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': None, 
        'rttm_filepath': None, 
        'uem_filepath' : None
    }
    with open(os.path.join(output_dir,'input_manifest.json'),'w') as fp:
        json.dump(meta,fp)
        fp.write('\n')


    MODEL_CONFIG = os.path.join(output_dir,'offline_diarization.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization.yaml"
        MODEL_CONFIG = wget.download(config_url,output_dir)

    config = OmegaConf.load(MODEL_CONFIG)
    # print(OmegaConf.to_yaml(config))

    """As can be seen most of the variables in config are self explanatory 
    with VAD variables under vad section and speaker related variables under speaker embeddings section.

    To perform VAD based diarization we can ignore `oracle_vad_manifest` in `speaker_embeddings` section for now and needs to fill up the rest. We also needs to provide pretrained `model_path` of vad and speaker embeddings .nemo models
    """

    pretrained_vad = 'vad_marblenet'
    pretrained_speaker_model = 'ecapa_tdnn'

    """Note in this tutorial, we use the VAD model MarbleNet-3x2 introduced and published in [ICASSP MarbleNet](https://arxiv.org/pdf/2010.13886.pdf). You might need to tune on dev set similar to your dataset if you would like to improve the performance.

    And the speakerNet-M-Diarization model achieves 7.3% confusion error rate on CH109 set with oracle vad. This model is trained on voxceleb1, voxceleb2, Fisher, SwitchBoard datasets. So for more improved performance specific to your dataset, finetune speaker verification model with a devset similar to your test set.
    """

    config.diarizer.manifest_filepath = os.path.join(output_dir,'input_manifest.json')
    config.diarizer.out_dir = output_dir #Directory to store intermediate files and prediction outputs

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
    config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
    config.diarizer.clustering.parameters.oracle_num_speakers=False

    #Here we use our inhouse pretrained NeMo VAD 
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.window_length_in_sec = 0.15
    config.diarizer.vad.shift_length_in_sec = 0.01
    config.diarizer.vad.parameters.onset = 0.8 
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.min_duration_on = 0.1
    config.diarizer.vad.parameters.min_duration_off = 0.4

    sd_model = ClusteringDiarizer(cfg=config)
    sd_model.diarize()

    output_path = os.path.join(output_dir, 'pred_rttms', wav_path.split('/')[-1].split(".wav")[0]+'.rttm')
    pred_labels = rttm_to_labels(output_path)
    return pred_labels
