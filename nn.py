#nn
import torch
import torch.nn as nn

#create a NN with the specified parameters in param
#params[0] = number of layers
#params[1] = optimizer
#params[2] = activator
def create_NN(params):
	return NN(params[0], params[1], params[2], params)

class NN(nn.Module):
	def __init__(self, num_layers, optimizer, activator, params):
		super(ConvNet, self).__init__()

		#keep track of own params
		self.params = params

		#set optimizer
		#can be one of the following: Adadelta, Adagrad, Adam, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD
		self.optimizer = optimizer

		#set activation function
		#can be one of the following: ELU, Hardshrink, Hardtanh, LeakyReLU, LogSigmoid, PReLU, ReLUm ReLU6, RReLU, SELU, CELU, Sigmoid, Softplus, Softshrink, Softsign, Tanh, TanhShrink, Threshold
		self.activator = activator

		#store layers in list
		self.layers = list()

		#we want to input a 28x28 images (784 values) and return an integer 0-9
		#to do this, we take 784 - 10 = 774 which factors into 2, 3, 6, 9
		#to make layer splitting easier, the only options for num_layers will be limited to these numbers
		if(num_layers == 2):	#774 / 2 = 387
			self.layers.append(nn.linear(784, 397))
			self.layers.append(nn.linear(397, 10))
		if(num_layers == 3):	#774 / 3 = 258
			self.layers.append(nn.linear(784, 526))
			self.layers.append(nn.linear(526, 268))
			self.layers.append(nn.linear(268, 10))
		if(num_layers == 6):	#774 / 6 = 129
			self.layers.append(nn.linear(784, 655))
			self.layers.append(nn.linear(655, 526))
			self.layers.append(nn.linear(526, 396))
			self.layers.append(nn.linear(396, 268))
			self.layers.append(nn.linear(268, 139))
			self.layers.append(nn.linear(139, 10))
		if(num_layers == 9):	#774 / 9 = 86
			self.layers.append(nn.linear(784, 698))
			self.layers.append(nn.linear(698, 612))
			self.layers.append(nn.linear(612, 526))
			self.layers.append(nn.linear(526, 440))
			self.layers.append(nn.linear(440, 354))
			self.layers.append(nn.linear(354, 268))
			self.layers.append(nn.linear(268, 182))
			self.layers.append(nn.linear(182, 96))
			self.layers.append(nn.linear(96, 10))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
