import random
import nn
import data

import torch
from torch.autograd import Variable

#create a NN with random parameters
def create_NN():
	#pick number of layers
	num_layers = random.choice([2, 3, 6, 9])

	#pick activation function
	actf_id = random.randint(0, 9)
	if(actf_id  == 0):
		actf = nn.nn.ELU()
	elif(actf_id  == 1):
		actf = nn.nn.Hardshrink()
	elif(actf_id  == 2):
		actf = nn.nn.LeakyReLU()
	elif(actf_id  == 3):
		actf = nn.nn.LogSigmoid()
	elif(actf_id  == 4):
		actf = nn.nn.PReLU()
	elif(actf_id  == 5):
		actf = nn.nn.ReLU()
	elif(actf_id  == 6):
		actf = nn.nn.ReLU6()
	elif(actf_id  == 7):
		actf = nn.nn.RReLU()
	elif(actf_id  == 8):
		actf = nn.nn.SELU()
	elif(actf_id  == 9):
		actf = nn.nn.CELU()

	#pick random optimizer
	optim_id = random.randint(0, 9)
	
	#create neural network and return
	return nn.NN(num_layers, actf, optim_id)

def train(model):
	#set up hyerparameters
	epochs = 5
	batch_size = 100
	learning_rate = 0.01

	#set up optimizer
	if(model.optimizer_id == 0):
		optimizer = nn.torch.optim.Adadelta(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 1):
		optimizer = nn.torch.optim.Adagrad(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 2):
		optimizer = nn.torch.optim.Adam(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 3):
		optimizer = nn.torch.optim.SparseAdam(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 4):
		optimizer = nn.torch.optim.Adamax(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 5):
		optimizer = nn.torch.optim.ASGD(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 6):
		optimizer = nn.torch.optim.LBFGS(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 7):
		optimizer = nn.torch.optim.RMSprop(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 8):
		optimizer = nn.torch.optim.Rprop(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 9):
		optimizer = nn.torch.optim.SGD(model.parameters(), lr = learning_rate)

	#set up loss function
	criterion = nn.nn.CrossEntropyLoss()

	#data loader
	train_loader = torch.utils.data.DataLoader(data.get_mnist_train(), batch_size = batch_size, shuffle = True)

	#training
	for epoch_id in range(epochs):
		for batch_id, (inps, vals) in enumerate(train_loader):
			#format inps
			inps.resize_(batch_size, 1, 784)

			#forward pass
			outs = model(inps)

			for i in range(784):
				print(inps[99][0][i], end = " ")
				if(i + 1 % 28 == 0):
					print()

			loss = criterion(outs, vals)

			#backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if(batch_id % 25 == 0):
				print("epoch: [{}/{}]; batch: [{}]; loss: {} ".format(epoch_id + 1, epochs, batch_id + 1, loss.item()))


net = create_NN()

train(net)