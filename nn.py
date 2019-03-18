#nn
import torch
import torch.nn as nn

class NN(nn.Module):
	def __init__(self, num_layers, activator_id, optimizer_id):
		super(NN, self).__init__()
		#track number of layers
		self.num_layers = num_layers

		#track activation function
		self.activator_id = activator_id

		#track optimizer id
		#can be one of the following: Adadelta, Adagrad, Adam, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD
		self.optimizer_id = optimizer_id

		#set activation function
		#can be one of the following: ELU, Hardshrink, LeakyReLU, LogSigmoid, PReLU, ReLU, ReLU6, RReLU, SELU, CELU, Sigmoid
		if(activator_id  == 0):
			self.activator = nn.ELU()
		elif(activator_id  == 1):
			self.activator = nn.Hardshrink()
		elif(activator_id  == 2):
			self.activator = nn.LeakyReLU()
		elif(activator_id  == 3):
			self.activator = nn.LogSigmoid()
		elif(activator_id  == 4):
			self.activator = nn.PReLU()
		elif(activator_id  == 5):
			self.activator = nn.ReLU()
		elif(activator_id  == 6):
			self.activator = nn.ReLU6()
		elif(activator_id  == 7):
			self.activator = nn.RReLU()
		elif(activator_id  == 8):
			self.activator = nn.SELU()
		elif(activator_id  == 9):
			self.activator = nn.CELU()

		#we want to input a 28x28 images (784 values) and return an integer 0-9
		#to do this, we take 784 - 10 = 774 which factors into 2, 3, 6, 9
		#to make layer splitting easier, the only options for num_layers will be limited to these numbers
		if(num_layers == 2):	#774 / 2 = 387
			self.layers = nn.Sequential(
			nn.Linear(784, 397),
			self.activator,
			nn.Linear(397, 10))
		elif(num_layers == 3):	#774 / 3 = 258
			self.layers = nn.Sequential(
			nn.Linear(784, 526),
			self.activator,
			nn.Linear(526, 268),
			self.activator,
			nn.Linear(268, 10))
		elif(num_layers == 6):	#774 / 6 = 129
			self.layers = nn.Sequential(
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
		elif(num_layers == 9):	#774 / 9 = 86
			self.layers = nn.Sequential(
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
		out = x.reshape(x.size(0), -1)
		return self.layers(out)
