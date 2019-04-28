#nn
import torch
import torch.nn as nn

class NN(nn.Module):
	def __init__(self, num_layers, neurons, activator_id, optimizer_id):
		super(NN, self).__init__()
		#track number of layers
		self.num_layers = num_layers

		#track number of neurons in each layer
		self.neurons = neurons

		#track activation function
		self.activator_id = activator_id

		#track optimizer id
		#can be one of the following: Adadelta, Adagrad, Adam, Adamax, ASGD, RMSprop, Rprop, SGD
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
		if(num_layers == 1):
			self.layers = nn.Sequential(
			nn.Linear(784, 10),
			self.activator)
		elif(num_layers == 2):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], 10),
			self.activator)
		elif(num_layers == 3):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], self.neurons[1]),
			self.activator,
			nn.Linear(self.neurons[1], 10),
			self.activator)
		elif(num_layers == 4):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], self.neurons[1]),
			self.activator,
			nn.Linear(self.neurons[1], self.neurons[2]),
			self.activator,
			nn.Linear(self.neurons[2], 10),
			self.activator)
		elif(num_layers == 5):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], self.neurons[1]),
			self.activator,
			nn.Linear(self.neurons[1], self.neurons[2]),
			self.activator,
			nn.Linear(self.neurons[2], self.neurons[3]),
			self.activator,
			nn.Linear(self.neurons[3], 10),
			self.activator)
		elif(num_layers == 6):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], self.neurons[1]),
			self.activator,
			nn.Linear(self.neurons[1], self.neurons[2]),
			self.activator,
			nn.Linear(self.neurons[2], self.neurons[3]),
			self.activator,
			nn.Linear(self.neurons[3], self.neurons[4]),
			self.activator,
			nn.Linear(self.neurons[4], 10),
			self.activator)
		elif(num_layers == 7):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], self.neurons[1]),
			self.activator,
			nn.Linear(self.neurons[1], self.neurons[2]),
			self.activator,
			nn.Linear(self.neurons[2], self.neurons[3]),
			self.activator,
			nn.Linear(self.neurons[3], self.neurons[4]),
			self.activator,
			nn.Linear(self.neurons[4], self.neurons[5]),
			self.activator,
			nn.Linear(self.neurons[5], 10),
			self.activator)
		elif(num_layers == 8):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], self.neurons[1]),
			self.activator,
			nn.Linear(self.neurons[1], self.neurons[2]),
			self.activator,
			nn.Linear(self.neurons[2], self.neurons[3]),
			self.activator,
			nn.Linear(self.neurons[3], self.neurons[4]),
			self.activator,
			nn.Linear(self.neurons[4], self.neurons[5]),
			self.activator,
			nn.Linear(self.neurons[5], self.neurons[6]),
			self.activator,
			nn.Linear(self.neurons[6], 10),
			self.activator)
		elif(num_layers == 9):
			self.layers = nn.Sequential(
			nn.Linear(784, self.neurons[0]),
			self.activator,
			nn.Linear(self.neurons[0], self.neurons[1]),
			self.activator,
			nn.Linear(self.neurons[1], self.neurons[2]),
			self.activator,
			nn.Linear(self.neurons[2], self.neurons[3]),
			self.activator,
			nn.Linear(self.neurons[3], self.neurons[4]),
			self.activator,
			nn.Linear(self.neurons[4], self.neurons[5]),
			self.activator,
			nn.Linear(self.neurons[5], self.neurons[6]),
			self.activator,
			nn.Linear(self.neurons[6], self.neurons[7]),
			self.activator,
			nn.Linear(self.neurons[7], 10),
			self.activator)

	def forward(self, x):
		out = x.reshape(x.size(0), -1)
		return self.layers(out)
