import torch
import import torch.nn as nn


def ContentLoss(nn.Module):
	def __init__(self,weight,loss_type):
		super(ContentLoss,self).__init__();
		self.weight  = weight;
		self.target = 0;
		self.output = 0;
		if loss_type == 'L2':
			self.crit = nn.MSELoss();
		if loss_type == 'L1':
			self.crit = nn.L1Loss();
		if loss_type == 'SmoothL1':
			self.crit = nn.SmoothL1Loss();
		else：
			print('dalong log : not supported now !');
			exit();
	def forward(self,input):
		if self.mode == 'capture':
			self.target = input.clone();
		elif self.mode == 'loss':
			self.loss = self.weight * self.crit(input,self.target);
		self.output = input;
		return self.output;


class PercepturalLoss(nn.Module):

	def __init__(self,args):
		super(PercepturalLoss,self.)..__init__();
		self.content_layers = args.content_layers;
		self.content_loss_layer = [];
		self.net = agrs.cnn;
		self.net.evaluate();
		for i ,layer_string in enumerate(self.content_layers):
			weight = args.content_weight[i];
			content_loss_layer = ContentLoss(weight,args.loss_type);
			layer_utils.insert_after(self.net,layer_string,content_loss_layer);
			self.content_loss_layer.append(content_loss_layer);

		layer_utils.trim_network(self.net);

