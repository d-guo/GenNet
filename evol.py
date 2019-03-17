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

	#create neural network
	net = nn.NN(num_layers, actf)

	#pick random optimizer
	optim_id = random.randint(0, 10)
	if(optim_id  == 0):
		optim = nn.torch.optim.Adadelta(net.parameters())
		net.optimizer = optim
	if(optim_id  == 1):
		optim = nn.torch.optim.Adagrad(net.parameters())
		net.optimizer = optim
	if(optim_id  == 2):
		optim = nn.torch.optim.Adam(net.parameters())
		net.optimizer = optim
	if(optim_id  == 3):
		optim = nn.torch.optim.SparseAdam(net.parameters())
		net.optimizer = optim
	if(optim_id  == 4):
		optim = nn.torch.optim.Adamax(net.parameters())
		net.optimizer = optim
	if(optim_id  == 5):
		optim = nn.torch.optim.ASGD(net.parameters())
		net.optimizer = optim
	if(optim_id  == 6):
		optim = nn.torch.optim.LBFGS(net.parameters())
		net.optimizer = optim
	if(optim_id  == 7):
		optim = nn.torch.optim.RMSprop(net.parameters())
		net.optimizer = optim
	if(optim_id  == 8):
		optim = nn.torch.optim.Rprop(net.parameters())
		net.optimizer = optim
	if(optim_id  == 9):
		optim = nn.torch.optim.SGD(net.parameters())
		net.optimizer = optim
	return net

foo = create_NN()