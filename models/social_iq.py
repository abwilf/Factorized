import sys; sys.path.append('/work/awilf/CMU-MultimodalSDK')
sys.path.append('/work/awilf/CMU-MultimodalSDK/mmsdk/mmmodelsdk/fusion')
import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
import torch
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy
import torch.optim as optim
import time
import scipy.misc
import os
import mmsdk
from mmsdk import mmdatasdk
from mmsdk.mmmodelsdk.fusion import TensorFusion
import numpy
import pickle
from random import shuffle
import time
from .mylstm import MyLSTM
from .densenet import get_densenet_features
from process_VAD import *


print ("Tensor-MFN code for Social-IQ")
print ("Yellow warnings fro SDK are ok!")
print ("If you do happen to get nans, then the reason is the most recent acoustic features update. You can replace nans and infs in acoustic at your discretion.")

metadata_template = { "root name": '', "computational sequence description": '', "computational sequence version": '', "alignment compatible": '', "dataset name": '', "dataset version": '', "creator": '', "contact": '', "featureset bib citation": '', "dataset bib citation": ''}
def get_compseq(path, key_name):
    if 'pk' in path:
        a = load_pk(path)
        compseq = mmdatasdk.computational_sequence(key_name)
        compseq.setData(a, key_name)
        metadata_template['root name'] = key_name
        compseq.setMetadata(metadata_template, key_name)
    else:
        assert 'csd' in path
        a = mmdatasdk.mmdataset({key_name: path})
        compseq = a[key_name]
    return compseq

def get_compseq_obj(obj, key_name):
    if type(obj) is dict:
        compseq = mmdatasdk.computational_sequence(key_name)
        compseq.setData(obj, key_name)
        metadata_template['root name'] = key_name
        compseq.setMetadata(metadata_template, key_name)
    else:
        compseq = obj[key_name]
    return compseq

def add_seq(dataset, obj, key_name, obj_type='path'):
    if obj_type == 'path':
        compseq = get_compseq(obj, key_name)
    else:
        compseq = get_compseq_obj(obj, key_name)
    dataset.computational_sequences[key_name] = compseq


def myavg(intervals,features):
    final=numpy.average(features,axis=0)
    if len(final.shape)==2:
        final=numpy.average(final,axis=0)
    return final

def raw_to_csd():
    # video to csd
    # if gc['video_feat'] == 'densenet':
    #     video_save_path = join(gc["csd_data"], f'{gc["video_feat"]}.pk')
    #     all_videos = join(gc['raw_data'], 'video')
    #     get_densenet_features(all_videos, desired_fps=1, temp_save_path=video_save_path)
    
    wav_dir = join(gc['raw_data'], 'audio')
    text_path = join(gc['raw_data'], 'text')

    ## MFA: vtt to aligned transcripts
    mfa_path = join(text_path, 'mfa')
    mkdirp(mfa_path)
    vtt_dir = text_path
    corpus_dir = join(mfa_path, 'corpus')
    aligned_dir = join(mfa_path, 'aligned')
    mfa_intervals_path = join(mfa_path, 'mfa_intervals.json')
    if not exists(mfa_intervals_path):
    # if True:
        import sys; sys.path.append('/work/awilf/mfa/'); from mfa_utils import main as mfa
        mfa(wav_dir, vtt_dir, corpus_dir, aligned_dir, mfa_intervals_path)
    
    ## Voice Activity Detection: from wavs to utterances
    vad_dir = join(text_path,'vad')
    mkdirp(vad_dir)
    vad_intervals_path = join(vad_dir, 'utterance_intervals.pk')
    # vad_intervals_path = '/work/awilf/MTAG2/data/vad_intervals.pk'
    if not exists(vad_intervals_path):
        get_vad(wav_dir, vad_dir, vad_intervals_path)
    
    ## Speaker Diarization: "squash" vad outputs together based on speaker
    speaker_diar_path = join(vad_dir, 'speaker_diar.pk')
    if not exists(speaker_diar_path):
        get_squashed(vad_intervals_path, speaker_diar_path)

    ## Split data based on diarization (speaker turn timings) and mfa output (words with timestamps)
    utt_split_path = join(vad_dir, 'utt_splits.pk')
    if not exists(utt_split_path):
        mfa = load_json(mfa_intervals_path)
        diar = load_pk(speaker_diar_path)
        split_words_speaker_turns(mfa, diar, utt_split_path)
    utt_splits = load_pk(utt_split_path)
    
    ## Get full BERT features from transcript, of form below.
    '''
    vid_id: [ 
    { # utterance 0
        'boundaries': (start,end), # within video
        'speaker_id': {0/1/2},
        'features': [1,n+2,768], # where n is the number of tokens in the utterance; [CLS] [*words tokens] [SEP]
        'intervals': [n,768],
        'words': str, original words in case you need them down the line.  Note: n may be different than len(words.split(' ')) because some words lead to multiple tokens (e.g. that's).
                We adjust the intervals to split up intervals evenly across these tokens in the above function.
    },
    ]
    '''
    full_bert_path = join(vad_dir, 'full_bert_feats.pk')
    if not exists(full_bert_path):
        bert_features = get_bert_features(utt_splits, full_bert_path)
    else:
        bert_features = load_pk(full_bert_path)

    ## Get CSD from BERT features
    transcript_csd_path = join(gc['csd_data'], 'transcript.pk')
    if not exists(transcript_csd_path):
        bert_features_to_csd(bert_features, transcript_csd_path)


def csd_to_processed():
    dataset = mmdatasdk.mmdataset(recipe={'dummy': join(gc['csd_data'], 'dummy.csd')})
    del dataset.computational_sequences['dummy']

    if gc['gran'] == 'chunk':
        # assert False, 'granularity must be word level for now'
        add_seq(dataset, join(gc["csd_data"], 'SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT.csd'), 'text')
    else:
        add_seq(dataset, join(gc["csd_data"], 'transcript.pk'), 'text')
    
    add_seq(dataset, join(gc["csd_data"], 'SOCIAL_IQ_COVAREP.csd'), 'audio')
    # add_seq(dataset, join(gc["csd_data"], 'SOCIAL_IQ_DENSENET161_1FPS.csd'), 'video')
    add_seq(dataset, join(gc["csd_data"], 'beit.pk'), 'video')

    dataset.align("text",collapse_functions=[myavg])
    dataset.impute("text")
    dataset.revert()
    dataset.unify()

    # add_seq(dataset, join(gc['csd_data'], "SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"), 'QA_BERT_lastlayer_binarychoice')
    add_seq(dataset, join(gc['csd_data'], "new_qa.pk"), 'QA_BERT_lastlayer_binarychoice')
    # add_seq(dataset, join(gc['csd_data'], "dummy.csd"), 'QA_BERT_lastlayer_binarychoice')
    
    return dataset

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

def get_judge():
    return nn.Sequential(OrderedDict([
        ('fc0',   nn.Linear(340,25)),
        ('sig0', nn.Sigmoid()),
        ('fc1',   nn.Linear(25,1)),
        ('sig1', nn.Sigmoid())
        ]))

def flatten_qail(_input):
    try:
        return _input.reshape(-1,*(_input.shape[3:])).squeeze().transpose(1,0)
    except:
        return _input.reshape(-1,*(_input.shape[3:])).squeeze().transpose(1,0,2)

def build_qa_binary(qa_glove,keys):
    return qai_to_tensor(qa_glove,keys,1)

def build_visual(visual,keys):
    vis_features=[]
    for i in range (len(keys)):
        this_vis=numpy.array(visual[keys[i]]["features"])
        this_vis=numpy.concatenate([this_vis,numpy.zeros([gc['seq_len'],768])],axis=0)[:gc['seq_len'],:]
        vis_features.append(this_vis)
    return numpy.array(vis_features,dtype="float32").transpose(1,0,2)

def build_acc(acoustic,keys):
    acc_features=[]
    for i in range (len(keys)):
        this_acc=numpy.array(acoustic[keys[i]]["features"])
        numpy.nan_to_num(this_acc)
        this_acc=numpy.concatenate([this_acc,numpy.zeros([gc['seq_len'],74])],axis=0)[:gc['seq_len'],:]
        acc_features.append(this_acc)
    final=numpy.array(acc_features,dtype="float32").transpose(1,0,2)
    return numpy.array(final,dtype="float32")

 
def build_trs(trs,keys):
    trs_features=[]
    for i in range (len(keys)):
        this_trs=numpy.array(trs[keys[i]]["features"][:,-768:])
        this_trs=numpy.concatenate([this_trs,numpy.zeros([gc['seq_len'],768])],axis=0)[:gc['seq_len'],:]
        trs_features.append(this_trs)
    return numpy.array(trs_features,dtype="float32").transpose(1,0,2)
 
gc = {}
social_iq = None
def process_data(keys, name, _gc):
    global gc,social_iq # need for seq_len; so we don't reprocess
    gc = _gc

    save_path = join(gc['proc_data'], f'{name}_social_{gc["gran"]}_{gc["seq_len"]}.pk')
    res = load_pk(save_path)
    if res is None:
    # if True:
        if social_iq is None:
            raw_to_csd()
            social_iq = csd_to_processed()
        label_keys = social_iq['QA_BERT_lastlayer_binarychoice'].keys()
        keys = [elt for elt in keys if elt in label_keys] # trim to label keys
        print(f'Building and writing processed data for {save_path}')
        qa_glove=social_iq["QA_BERT_lastlayer_binarychoice"]
        visual=social_iq["video"]
        transcript=social_iq["text"]
        acoustic=social_iq["audio"]
        
        if gc['gran'] != 'chunk':
            keys = [elt for elt in keys if elt in transcript.keys()] # filter b/c missing some word-level

        qas=build_qa_binary(qa_glove,keys)
        visual=build_visual(visual,keys)
        trs=build_trs(transcript,keys)	
        acc=build_acc(acoustic,keys)
        intervals=[numpy.array(social_iq["text"][key]['intervals'])[:gc['seq_len']] for key in keys]
        res = qas,visual,trs,acc,keys,intervals
        save_pk(save_path, res)
    else:
        print('Loading processed data')
    return res

def to_pytorch(_input):
    return Variable(torch.tensor(_input)).cuda()

def reshape_to_correct(_input,shape):
    return _input[:,None,None,:].expand(-1,shape[1],shape[2],-1).reshape(-1,_input.shape[1])

def calc_accuracy(correct,incorrect):
    correct_=correct.cpu()
    incorrect_=incorrect.cpu()
    return numpy.array(correct_>incorrect_,dtype="float32").sum()/correct.shape[0]

def feed_forward(keys,q_lstm,a_lstm,v_lstm,t_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn,preloaded_data=None):
    q,a,i=[data[keys[0]:keys[1]] for data in preloaded_data[0]]
    vis=preloaded_data[1][:,keys[0]:keys[1],:]
    trs=preloaded_data[2][:,keys[0]:keys[1],:]
    acc=preloaded_data[3][:,keys[0]:keys[1],:]

    reference_shape=q.shape
    q_rep=q_lstm.step(to_pytorch(flatten_qail(q)))[1][0][0,:,:]
    a_rep=a_lstm.step(to_pytorch(flatten_qail(a)))[1][0][0,:,:]
    i_rep=a_lstm.step(to_pytorch(flatten_qail(i)))[1][0][0,:,:]

    #transcript representation
    t_full=t_lstm.step(to_pytorch(trs))
    #visual representation
    v_full=v_lstm.step(to_pytorch(vis))
    #acoustic representation
    ac_full=ac_lstm.step(to_pytorch(acc))

    t_seq=t_full[0]
    v_seq=v_full[0]
    ac_seq=ac_full[0]

    t_rep_extended=reshape_to_correct(t_full[1][0][0,:,:],reference_shape)
    v_rep_extended=reshape_to_correct(v_full[1][0][0,:,:],reference_shape)
    ac_rep_extended=reshape_to_correct(ac_full[1][0][0,:,:],reference_shape)
    

    #MFN and TFN Dance! 
    before_tfn=torch.cat([mfn_delta2((mfn_delta1(torch.cat([t_seq[i],t_seq[i+1],v_seq[i],v_seq[i+1],ac_seq[i],ac_seq[i+1]],dim=1))*torch.cat([t_seq[i],t_seq[i+1],v_seq[i],v_seq[i+1],ac_seq[i],ac_seq[i+1]],dim=1)))[None,:,:] for i in range(t_seq.shape[0]-1)],dim=0)
    after_tfn=torch.cat([mfn_tfn.fusion([before_tfn[i,:,:50],before_tfn[i,:,50:70],before_tfn[i,:,70:]])[None,:,:] for i in range(t_seq.shape[0]-1)],dim=0)
    after_mfn=mfn_mem.step(after_tfn)[1][0][0,:,:]
    mfn_final=reshape_to_correct(after_mfn,reference_shape)
    
    return q_rep,a_rep,i_rep,t_rep_extended,v_rep_extended,ac_rep_extended,mfn_final

def init_tensor_mfn_modules():
    q_lstm=MyLSTM(768,50).cuda()
    a_lstm=MyLSTM(768,50).cuda()
    t_lstm=MyLSTM(768,50).cuda()
    v_lstm=MyLSTM(768,20).cuda()
    ac_lstm=MyLSTM(74,20).cuda()

    mfn_mem=MyLSTM(100,100).cuda()
    mfn_delta1=nn.Sequential(OrderedDict([
            ('fc0',   nn.Linear(180,25)),
            ('relu0', nn.ReLU()),
            ('fc1',   nn.Linear(25,180)),
            ('relu1', nn.Softmax())
            ])).cuda()

    mfn_delta2=nn.Sequential(OrderedDict([
                ('fc0',   nn.Linear(180,90)),
                ('relu0', nn.ReLU()),
                ])).cuda()


    mfn_tfn=TensorFusion([50,20,20],100).cuda()
    return q_lstm,a_lstm,t_lstm,v_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn

def replace_inf(arr):
    arr[arr==-numpy.inf]=numpy.isfinite(arr).min()
    arr[arr==numpy.inf]=numpy.isfinite(arr).max()
    return arr

def get_train_dev():
    #if you have enough RAM, specify this as True - speeds things up ;)
    preload=True
    bs=32
    trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
    #This video has some issues in training set
    bads=['f5NJQiY9AuY','aHBLOkfJSYI']
    folds=[trk,dek]
    for bad in bads:
        for fold in folds:
            try:
                fold.remove(bad)
            except:
                pass

    q_lstm,a_lstm,t_lstm,v_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn=init_tensor_mfn_modules()
    preloaded_train=process_data(trk)
    preloaded_dev=process_data(dek)
    
    # if preload is True:
    # 	if not os.path.exists('train.pk'):
    # 		save_pk('train.pk', preloaded_train)
    # 		save_pk('dev.pk', preloaded_dev)
        
    # 	else:
    # 		preloaded_train = load_pk('train.pk')
    # 		preloaded_dev = load_pk('dev.pk')

    # else:
    # 	preloaded_data=None

    replace_inf(preloaded_train[3])
    replace_inf(preloaded_dev[3])

    return preloaded_train, preloaded_dev
    
if __name__=="__main__":


    preloaded_train, preloaded_dev = get_train_dev()
    
    #Getting the Judge
    judge=get_judge().cuda()

    #Initializing parameter optimizer
    params=	list(q_lstm.parameters())+list(a_lstm.parameters())+list(judge.parameters())+\
        list(t_lstm.parameters())+list(v_lstm.parameters())+list(ac_lstm.parameters())+\
        list(mfn_mem.parameters())+list(mfn_delta1.parameters())+list(mfn_delta2.parameters())+list(mfn_tfn.linear_layer.parameters())

    optimizer=optim.Adam(params,lr=0.001)


    for i in range (100):
        print ("Epoch %d"%i)
        losses=[]
        accs=[]
        ds_size=len(trk)
        for j in range(int(ds_size/bs)+1):

            if preload is True:
                this_trk=[j*bs,(j+1)*bs]
            else:
                this_trk=trk[j*bs:(j+1)*bs]

            q_rep,a_rep,i_rep,v_rep,t_rep,ac_rep,mfn_rep=feed_forward(this_trk,q_lstm,a_lstm,v_lstm,t_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn,preloaded_train)
                
            real_bs=float(q_rep.shape[0])

            correct=judge(torch.cat((q_rep,a_rep,i_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
            incorrect=judge(torch.cat((q_rep,i_rep,a_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
    
            correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
            incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
            
            optimizer.zero_grad()
            loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean))
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            accs.append(calc_accuracy(correct,incorrect))
            
        print ("Loss %f",numpy.array(losses,dtype="float32").mean())
        print ("Accs %f",numpy.array(accs,dtype="float32").mean())


        _accs=[]
        ds_size=len(dek)
        for j in range(int(ds_size/bs)+1):

            if preload is True:
                this_dek=[j*bs,(j+1)*bs]
            else:
                this_dek=trk[j*bs:(j+1)*bs]


            q_rep,a_rep,i_rep,v_rep,t_rep,ac_rep,mfn_rep=feed_forward(this_dek,q_lstm,a_lstm,v_lstm,t_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn,preloaded_dev)

            real_bs=float(q_rep.shape[0])


            correct=judge(torch.cat((q_rep,a_rep,i_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
            incorrect=judge(torch.cat((q_rep,i_rep,a_rep,t_rep,v_rep,ac_rep,mfn_rep),1))

            correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
            incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
            
            _accs.append(calc_accuracy(correct,incorrect))
            
        print ("Dev Accs %f",numpy.array(_accs,dtype="float32").mean())
        print ("-----------")

