import nn
import evol
import matplotlib.pyplot as plt
import data

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
			else:
				print("Actual number: {}".format(vals[0].item()))
				print("Predicted number: {}".format(pred[0].item()))
				plt.imshow(inps[0][0])
				plt.show()


	return correct / total


file = open("gennetparams", "r")
lines = file.readlines()
p1 = int(lines[0])
p2 = int(lines[1])
p3 = int(lines[2])

model = nn.NN(p1, p2, p3)
model.load_state_dict(nn.torch.load("gennet"))

model.eval()

test(model)




"""
evol.evolve(20, 10, 0.2)



file = open("gennetparams", "r")
lines = file.readlines()
p1 = int(lines[0])
p2 = int(lines[1])
p3 = int(lines[2])

model = nn.NN(p1, p2, p3)
model.load_state_dict(nn.torch.load("gennet"))

model.eval()

test_data = data.get_mnist_test()

test_loader = nn.torch.utils.data.DataLoader(test_data, 1, shuffle = True)

train_iter = iter(test_loader)
images, labels = train_iter.next()
image = images[0][0]

output = model(images)
pred = output.data.max(1)[1]
print("Actual Num: {}".format(labels[0]))
print("Predicted Num: {}".format(pred[0]))

plt.imshow(image)
plt.show()

"""