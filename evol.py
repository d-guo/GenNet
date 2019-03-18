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

	#pick random optimizer
	optim_id = random.randint(0, 9)
	
	#create neural network and return
	return nn.NN(num_layers, actf_id, optim_id)

def train(model):
	#configure devices
	if(torch.cuda.is_available()):
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')

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
	loss_func = nn.nn.CrossEntropyLoss()

	#data loader
	train_loader = torch.utils.data.DataLoader(data.get_mnist_train(), batch_size = batch_size, shuffle = True)

	#training
	for epoch_id in range(epochs):
		for batch_id, (inps, vals) in enumerate(train_loader):
			#format inps and vals
			inps = inps.to(device)
			vals = vals.to(device)

			#forward pass
			outs = model(inps)
			loss = loss_func(outs, vals)

			#backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if(batch_id % 25 == 0):
				print("epoch: [{}/{}]; batch: [{}]; loss: {} ".format(epoch_id + 1, epochs, batch_id + 1, loss.item()))


def test_model_performance(model):
	#configure devices
	if(torch.cuda.is_available()):
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')

	#set up hyerparameters
	epochs = 5
	batch_size = 100

	#data loader
	test_loader = torch.utils.data.DataLoader(data.get_mnist_test(), batch_size = 1, shuffle = True)

	#testing
	with torch.no_grad():
		correct = 0
		total = 0
		for batch_id, (inps, vals) in enumerate(test_loader):
			#format inps and vals
			inps = inps.to(device)
			vals = vals.to(device)

			#forward pass
			outs = model(inps)
			pred = outs.data.max(1)[1]

			#increment accordingly
			total += 1
			if(pred[0].item() == vals[0].item()):
				correct += 1

	return correct / total



def save_model(model, model_name):
	torch.save(model.state_dict(), "{}".format(model_name))
	file = open("{}params".format(model_name), "w+")
	file.write(str(model.num_layers) + "\n")
	file.write(str(model.activator_id) + "\n")
	file.write(str(model.optimizer_id) + "\n")
	file.close()

"""
net = nn.NN(3, 5, 2)
train(net)
save_model(net, "test")
"""