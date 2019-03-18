import random
import nn

#create a NN with random parameters
def create_NN():
	#pick number of layers
	num_layers = random.choice([2, 3, 6, 9])

	#pick activation function
	actf_id = random.randint(0, 10)
	if(actf_id  == 0):
		actf = nn.nn.ELU()
	if(actf_id  == 1):
		actf = nn.nn.Hardshrink()
	if(actf_id  == 2):
		actf = nn.nn.LeakyReLU()
	if(actf_id  == 3):
		actf = nn.nn.LogSigmoid()
	if(actf_id  == 4):
		actf = nn.nn.PReLU()
	if(actf_id  == 5):
		actf = nn.nn.ReLU()
	if(actf_id  == 6):
		actf = nn.nn.ReLU6()
	if(actf_id  == 7):
		actf = nn.nn.RReLU()
	if(actf_id  == 8):
		actf = nn.nn.SELU()
	if(actf_id  == 9):
		actf = nn.nn.CELU()

	#pick random optimizer
	optim_id = random.randint(0, 10)
	
	#create neural network and return
	net = nn.NN(num_layers, actf, optim_id)

foo = create_NN()