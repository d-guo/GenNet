#nn
import torch
import torch.nn as nn

class NN(nn.Module):
	def __init__(self, num_layers, activator, optimizer_id):
		super(NN, self).__init__()

		#set activation function
		#can be one of the following: ELU, Hardshrink, LeakyReLU, LogSigmoid, PReLU, ReLU, ReLU6, RReLU, SELU, CELU, Sigmoid
		self.activator = activator

		#track optimizer id
		#can be one of the following: Adadelta, Adagrad, Adam, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD
		self.optimizer_id = optimizer_id

		#store layers in list
		self.layers = nn.ParameterList()

		#we want to input a 28x28 images (784 values) and return an integer 0-9
		#to do this, we take 784 - 10 = 774 which factors into 2, 3, 6, 9
		#to make layer splitting easier, the only options for num_layers will be limited to these numbers
		if(num_layers == 2):	#774 / 2 = 387
			self.layers = nn.Sequential(
			nn.Linear(784, 397),
			self.activator,
			nn.Linear(397, 10))
		if(num_layers == 3):	#774 / 3 = 258
			self.layers = nn.Sequential(
			nn.Linear(784, 526),
			self.activator,
			nn.Linear(526, 268),
			self.activator,
			nn.Linear(268, 10))
		if(num_layers == 6):	#774 / 6 = 129
			self.layers == nn.Sequential(
			nn.Linear(784, 655),
			self.activator,
			nn.Linear(655, 526),
			self.activator,
			nn.Linear(526, 396),
			self.activator,
			nn.Linear(396, 268),
			self.activator,
			nn.Linear(268, 139),
			self.activator,
			nn.Linear(139, 10))
			
		if(num_layers == 9):	#774 / 9 = 86
			self.layers == nn.Sequential(
			nn.Linear(784, 698),
			self.activator,
			nn.Linear(698, 612),
			self.activator,
			nn.Linear(612, 526),
			self.activator,
			nn.Linear(526, 440),
			self.activator,
			nn.Linear(440, 354),
			self.activator,
			nn.Linear(354, 268),
			self.activator,
			nn.Linear(268, 182),
			self.activator,
			nn.Linear(182, 96),
			self.activator,
			nn.Linear(96, 10))

	def forward(self, x):
		return self.layers(x)
