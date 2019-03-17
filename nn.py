#nn
import torch
import torch.nn as nn

class NN(nn.Module):
	def __init__(self, num_layers, activator):
		super(NN, self).__init__()

		#set activation function
		#can be one of the following: ELU, Hardshrink, LeakyReLU, LogSigmoid, PReLU, ReLU, ReLU6, RReLU, SELU, CELU, Sigmoid
		self.activator = activator

		#store layers in list
		self.layers = nn.ParameterList()

		#we want to input a 28x28 images (784 values) and return an integer 0-9
		#to do this, we take 784 - 10 = 774 which factors into 2, 3, 6, 9
		#to make layer splitting easier, the only options for num_layers will be limited to these numbers
		if(num_layers == 2):	#774 / 2 = 387
			self.layers.append(nn.Linear(784, 397))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(397, 10))
		if(num_layers == 3):	#774 / 3 = 258
			self.layers.append(nn.Linear(784, 526))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(526, 268))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(268, 10))
		if(num_layers == 6):	#774 / 6 = 129
			self.layers.append(nn.Linear(784, 655))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(655, 526))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(526, 396))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(396, 268))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(268, 139))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(139, 10))
		if(num_layers == 9):	#774 / 9 = 86
			self.layers.append(nn.Linear(784, 698))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(698, 612))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(612, 526))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(526, 440))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(440, 354))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(354, 268))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(268, 182))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(182, 96))
			self.layers.append(self.activator)
			self.layers.append(nn.Linear(96, 10))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
