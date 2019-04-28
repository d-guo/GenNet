import nn
import evol
import matplotlib.pyplot as plt
import data


evol.evolve(20, 10, 0.2)


def test(model):
	#configure devices
	if(nn.torch.cuda.is_available()):
		device = nn.torch.device('cuda:0')
	else:
		device = nn.torch.device('cpu')

	model.eval()

	#set up hyerparameters
	epochs = 5
	batch_size = 1

	#data loader
	test_loader = nn.torch.utils.data.DataLoader(data.get_mnist_test(), batch_size = batch_size, shuffle = True)

	#testing
	with nn.torch.no_grad():
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
			print("Actual number: {}".format(vals[0].item()))
			print("Predicted number: {}".format(pred[0].item()))
			plt.imshow(inps[0][0])
			plt.show()


	print(correct / total)

"""
file = open("gennetparams", "r")
lines = file.readlines()
p1 = int(lines[0])
neurons = list()
for i in range(p1 - 1):
	neurons.append(int(lines[i + 1]))
p2 = int(lines[p1])
p3 = int(lines[p1 + 1])

model = nn.NN(p1, neurons, p2, p3)
model.load_state_dict(nn.torch.load("gennet"))

model.eval()

test(model)
"""