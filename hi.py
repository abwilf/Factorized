from alex_utils import *
import sys
sys.path.append('/work/awilf/CMU-MultimodalSDK')
sys.path.append('/work/awilf/CMU-MultimodalSDK/mmsdk/mmmodelsdk/fusion')

import numpy
def myavg(intervals,features):
        return numpy.average(features,axis=0)

from mmsdk import mmdatasdk
# cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
# cmumosi_highlevel.align('glove_vectors',collapse_functions=[myavg])
# cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
# cmumosi_highlevel.align('Opinion Segment Labels')

# from mmsdk import mmdatasdk
# socialiq_highlevel=mmdatasdk.mmdataset(mmdatasdk.socialiq.highlevel,'/work/awilf/Social-IQ/socialiq/')
socialiq_highlevel=mmdatasdk.mmdataset('/work/awilf/Social-IQ/socialiq/')
# folds=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold

# socialiq_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'socialiq/')
# socialiq_highlevel.align('Opinion Segment Labels')

def qai_to_tensor(in_put,keys,total_i=1):
    data=dict(in_put.data)
    features=[]
    for i in range (len(keys)):
        features.append(numpy.array(data[keys[i]]["features"]))
    input_tensor=numpy.array(features,dtype="float32")[:,0,...]
    in_shape=list(input_tensor.shape)
    q_tensor=input_tensor[:,:,:,0:1,:,:]
    ai_tensor=input_tensor[:,:,:,1:,:,:]

    return q_tensor,ai_tensor[:,:,:,0:1,:,:],ai_tensor[:,:,:,1:1+total_i,:,:]

def align():
    #first time dl
    #socialiq_no_align=mmdatasdk.mmdataset(mmdatasdk.socialiq.highlevel,"socialiq")
    #second time dl
    socialiq_no_align=mmdatasdk.mmdataset("/work/awilf/Social-IQ/socialiq")
    # print(socialiq_no_align.computational_sequences.keys())
    # exit()
    #don't need these guys for aligning
    del socialiq_no_align.computational_sequences["b'SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE'"]
    del socialiq_no_align.computational_sequences["b'SOCIAL-IQ_QA_BERT_MULTIPLE_CHOICE'"]
    del socialiq_no_align.computational_sequences["b'SOCIAL_IQ_VGG_1FPS'"]
    socialiq_no_align.align("b'SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT'",collapse_functions=[myavg])
    #simple name change - now the dataset is aligned
    socialiq_aligned=socialiq_no_align
    
    socialiq_aligned.impute("b'SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT'")
    socialiq_aligned.revert()
    
    deploy_files={x:x for x in socialiq_aligned.keys()}
    socialiq_aligned.deploy("./deployed",deploy_files)
    return socialiq_aligned

# socialiq_aligned = align()
paths={}
paths["QA_BERT_lastlayer_binarychoice"]="/work/awilf/MTAG/deployed/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
paths["DENSENET161_1FPS"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_DENSENET161_1FPS'.csd"
paths["Transcript_Raw_Chunks_BERT"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT'.csd"
paths["Acoustic"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_COVAREP'.csd"
social_iq=mmdatasdk.mmdataset(paths)
social_iq.unify()

a = load_pk('vad_intervals.pk')
a = {k: np.array([ [np.float32(elt2) for elt2 in elt.replace('speaker_', '').split(' ')] for elt in v]) for k,v in a.items() if v is not None}
# vid = '-0REX0yx4QA'

# hi=2


wav_key = '1MwN5nDajWs'
np.array(social_iq['Transcript_Raw_Chunks_BERT'][f'{wav_key}']['intervals'])
a[f'{wav_key}_trimmed-out.wav']





