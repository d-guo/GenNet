import random
import nn
import data

import torch
from torch.autograd import Variable

#create a NN with random parameters
def create_NN():
	#pick number of layers
	num_layers = random.randint(1, 9)

	#pick neurons in each layer
	neurons = list()
	for i in range(num_layers - 1):
		neurons.append(random.randint(80, 1000))

	#pick activation function
	actf_id = random.randint(0, 9)

	#pick random optimizer
	optim_id = random.randint(0, 7)
	
	#create neural network and return
	return nn.NN(num_layers, neurons, actf_id, optim_id)

#save model and its parameters to be loaded later
def save_model(model, model_name):
	torch.save(model.state_dict(), "{}".format(model_name))
	file = open("{}params".format(model_name), "w+")
	file.write(str(model.num_layers) + "\n")
	for i in range(model.num_layers - 1):
		file.write(str(model.neurons[i]) + "\n")
	file.write(str(model.activator_id) + "\n")
	file.write(str(model.optimizer_id))
	file.close()

#trains an individual model
def train(model):
	#configure devices
	if(torch.cuda.is_available()):
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')

	#set up hyerparameters
	epochs = 1
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
		optimizer = nn.torch.optim.Adamax(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 4):
		optimizer = nn.torch.optim.ASGD(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 5):
		optimizer = nn.torch.optim.RMSprop(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 6):
		optimizer = nn.torch.optim.Rprop(model.parameters(), lr = learning_rate)
	elif(model.optimizer_id == 7):
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

	return model

#tests individual model performance
def test_model_performance(model):
	#configure devices
	if(torch.cuda.is_available()):
		device = torch.device('cuda:0')
	else:
		device = torch.device('cpu')

	model.eval()

	#set up hyerparameters
	epochs = 5
	batch_size = 1

	#data loader
	test_loader = torch.utils.data.DataLoader(data.get_mnist_test(), batch_size = batch_size, shuffle = True)

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

#mutates one parameter of the model
def mutate(model):
	key1 = random.random()
	#complete mutation 10% of the itme
	if(key1 <= 0.1):
		return create_NN()
	#otherwise just mutate one parameter
	else:
		key2 = random.randint(0, 2)
		if(key2 == 0):
		#change number of neurons in a random layer
			layer_num = randint(0, model.num_layers - 2)
			model.neurons[layer_num] = random.randint(80, 1000)
			return nn.NN(model.num_layers, model.neurons, model.activator_id, model.optimizer_id)
		elif(key2 == 1):
		#change activator
			return nn.NN(model.num_layers, model.neurons, random.randint(0, 9), model.optimizer_id)
		elif(key2 == 2):
		#change optimizer
			return nn.NN(model.num_layers, model.neurons, model.activator_id, random.randint(0, 7))

#creates child using parameters from two parents
def breed(m1, m2):
	#choose keys to determine which parent to take parameter from
	key1 = random.randint(0, 1)
	key2 = random.randint(0, 1)
	key3 = random.randint(0, 1)

	if(key1 == 0):
		num_layers = m1.num_layers
		neurons = m1.neurons
	else:
		num_layers = m2.num_layers
		neurons = m2.neurons
	if(key2 == 0):
		activator_id = m1.activator_id
	else:
		activator_id = m2.activator_id
	if(key3 == 0):
		optimizer_id = m1.optimizer_id
	else:
		optimizer_id = m2.optimizer_id

	return nn.NN(num_layers, neurons, activator_id, optimizer_id)

#creates initial population for training
def initial_pop(pop_num):
	#store population in list
	pop = list()

	for i in range(pop_num):
		pop.append(create_NN())

	return pop

#prevent double training
def reset_pop(pop):
	#new population without any training
	fresh_pop = list()

	#create new NN for each member of population
	for mem in pop:
		fresh_pop.append(nn.NN(mem.num_layers, mem.neurons, mem.activator_id, mem.optimizer_id))

	return fresh_pop

#actual evolution
def evolve(pop_size, num_gens, chance_of_mutation):
	#create initial population
	pop = initial_pop(pop_size)

	#evolution through generations
	for gen_id in range(num_gens):
		print("Generation {}".format(gen_id))
		#train each of the members of the generation
		for mem_id in range(pop_size):
			print("training member {} ({}, {}, {}, {})".format(mem_id, pop[mem_id].num_layers, pop[mem_id].neurons, pop[mem_id].activator_id, pop[mem_id].optimizer_id))
			pop[mem_id] = train(pop[mem_id])

		#get performance from each of the members and remove those below averge
		#store performance score in list
		perf_scores = list()
		for mem_id in range(len(pop)):
			print("testing member {}".format(mem_id))
			perf_scores.append(test_model_performance(pop[mem_id]))

		#print progress of current population
		print("Performance Summary of Generation {}:".format(gen_id))
		for i in range(pop_size):
			print("Member {} achieved accuracy of {} with parameters ({}, {}, {}, {})".format(i, perf_scores[i], pop[i].num_layers, pop[i].neurons, pop[i].activator_id, pop[i].optimizer_id))

		#don't do the following for the last generation
		if(gen_id != num_gens - 1):
			#find average and keep those above it
			average_perf = sum(perf_scores) / pop_size
			print("Average accuracy was {}".format(average_perf))

			new_pop = list()
			surv_inds = list()

			print("Surviving members:")
			for i in range(pop_size):
				if(perf_scores[i] >= average_perf):
					new_pop.append(pop[i])
					#track indices of surviving members for printing
					surv_inds.append(i)
					print("member {} survived".format(i))

			pop = new_pop.copy()

			#fill in remaining population members by breeding members of population
			while(len(pop) < pop_size):
				#pick two random parents
				m1_id = random.randint(0, len(new_pop) - 1)
				m2_id = random.randint(0, len(new_pop) - 1)

				#breed together if parents are not the same member
				if(m1_id != m2_id):
					new_NN = breed(new_pop[m1_id], new_pop[m2_id])

					#chance to mutate new child
					key = random.random()
					if(key <= chance_of_mutation):
						new_NN = mutate(new_NN)
						print("bred together members {} and {}, child mutated".format(surv_inds[m1_id], surv_inds[m2_id]))
					else:
						print("bred together members {} and {}, child not mutated".format(surv_inds[m1_id], surv_inds[m2_id]))

					pop.append(new_NN)

			#reset populations
			pop = reset_pop(pop)
			print()
		else:
			#save best performer after all generations
			best_perf_score = -1
			best_perf_id = -1

			for i in range(pop_size):
				if(perf_scores[i] > best_perf_score):
					best_perf_score = perf_scores[i]
					best_perf_id = i

				print("Best performer was member {} with an accuracy of {}".format(best_perf_id, best_perf_score))
				save_model(pop[best_perf_id], "gennet")
				return pop[best_perf_id]