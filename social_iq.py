import sys
sys.path.append('/work/awilf/CMU-MultimodalSDK')
sys.path.append('/work/awilf/CMU-MultimodalSDK/mmsdk/mmmodelsdk/fusion')
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
import mylstm
import mmsdk
from mmsdk import mmdatasdk
from mmsdk.mmmodelsdk.fusion import TensorFusion
import numpy
import pickle
from random import shuffle
import time


print ("Tensor-MFN code for Social-IQ")
print ("Yellow warnings fro SDK are ok!")
print ("If you do happen to get nans, then the reason is the most recent acoustic features update. You can replace nans and infs in acoustic at your discretion.")


#Loading the data of Social-IQ
# #Yellow warnings fro SDK are ok!
# if os.path.isdir("/work/awilf/MTAG/deployed/") is False:
# 	print ("Need to run the modality alignment first")
# 	exit()
# 	from alignment import align,myavg
# 	align()
 
paths={}
paths["QA_BERT_lastlayer_binarychoice"]="/work/awilf/Social-IQ/socialiq/SOCIAL-IQ_QA_BERT_LASTLAYER_BINARY_CHOICE.csd"
paths["DENSENET161_1FPS"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_DENSENET161_1FPS'.csd"
paths["Transcript_Raw_Chunks_BERT"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_TRANSCRIPT_RAW_CHUNKS_BERT'.csd"
paths["Acoustic"]="/work/awilf/MTAG/deployed/b'SOCIAL_IQ_COVAREP'.csd"
social_iq=mmdatasdk.mmdataset(paths)
social_iq.unify() 


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
		('fc0',   nn.Linear(214,25)),
		('sig0', nn.Sigmoid()),
		('fc1',   nn.Linear(25,1)),
		('sig1', nn.Sigmoid())
		]))

def flatten_qail(_input):
	return _input.reshape(-1,*(_input.shape[3:])).squeeze().transpose(1,0)
	

def build_qa_binary(qa_glove,keys):
	return qai_to_tensor(qa_glove,keys,1)


def build_visual(visual,keys):
	vis_features=[]
	for i in range (len(keys)):
		this_vis=numpy.array(visual[keys[i]]["features"])
		this_vis=numpy.concatenate([this_vis,numpy.zeros([25,2208])],axis=0)[:25,:]
		vis_features.append(this_vis)
	return numpy.array(vis_features,dtype="float32").transpose(1,0,2)

def build_acc(acoustic,keys):
	acc_features=[]
	for i in range (len(keys)):
		this_acc=numpy.array(acoustic[keys[i]]["features"])
		numpy.nan_to_num(this_acc)
		this_acc=numpy.concatenate([this_acc,numpy.zeros([25,74])],axis=0)[:25,:]
		acc_features.append(this_acc)
	final=numpy.array(acc_features,dtype="float32").transpose(1,0,2)
	return numpy.array(final,dtype="float32")

 
def build_trs(trs,keys):
	trs_features=[]
	for i in range (len(keys)):
		this_trs=numpy.array(trs[keys[i]]["features"][:,-768:])
		this_trs=numpy.concatenate([this_trs,numpy.zeros([25,768])],axis=0)[:25,:]
		trs_features.append(this_trs)
	return numpy.array(trs_features,dtype="float32").transpose(1,0,2)
 
def process_data(keys, name):
	save_path = f'/work/awilf/MTAG/{name}_social.pk'
	res = load_pk(save_path)
	if res is None:
		print(f'Building and writing processed data for {save_path}')
		qa_glove=social_iq["QA_BERT_lastlayer_binarychoice"]
		visual=social_iq["DENSENET161_1FPS"]
		transcript=social_iq["Transcript_Raw_Chunks_BERT"]
		acoustic=social_iq["Acoustic"]

		qas=build_qa_binary(qa_glove,keys)
		visual=build_visual(visual,keys)
		trs=build_trs(transcript,keys)	
		acc=build_acc(acoustic,keys)

		res = qas,visual,trs,acc
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
	q_lstm=mylstm.MyLSTM(768,50).cuda()
	a_lstm=mylstm.MyLSTM(768,50).cuda()
	t_lstm=mylstm.MyLSTM(768,50).cuda()
	v_lstm=mylstm.MyLSTM(2208,20).cuda()
	ac_lstm=mylstm.MyLSTM(74,20).cuda()

	mfn_mem=mylstm.MyLSTM(100,100).cuda()
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
	
import pickle
def load_pk(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        return load_pk_old(filename)

def load_pk_old(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

def save_pk(file_stub, pk, protocol=4):
    with open(file_stub, 'wb') as f:
        pickle.dump(pk, f, protocol=protocol)

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

