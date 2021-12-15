import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time
import scipy.misc
import os

class MyLSTM(nn.Module):

	def __init__(self,input_dim,output_dim,num_layers=1):
		super (MyLSTM,self).__init__()
		self.input_dim=input_dim
		self.output_dim=output_dim
		self.num_layers=num_layers
		self.rnn=nn.LSTM(input_dim,output_dim,num_layers)	
	def step(self,inputs):
		bs=inputs.shape[1]
		
		h0 = Variable(torch.zeros(self.num_layers, bs, self.output_dim)).cuda()
		c0 = Variable(torch.zeros(self.num_layers, bs, self.output_dim)).cuda()
		outs,hcn=self.rnn(inputs,(h0,c0))
		return outs,hcn


if __name__=='__main__':
	
	
	obj=MyLSTM(300,100,2).cuda()
	inputs = Variable(torch.randn(10, 20, 300)).cuda()
	out,hcn=obj.step(inputs)
	print(out.shape,hcn[0].shape)
	
