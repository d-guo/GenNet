import nn
import evol
import matplotlib.pyplot as plt
import data

print("SS")

file = open("testparams", "r")
lines = file.readlines()
p1 = int(lines[0])
p2 = int(lines[1])
p3 = int(lines[2])

model = nn.NN(p1, p2, p3)
model.load_state_dict(nn.torch.load("test"))

model.eval()

print(evol.test_model_performance(model))



"""


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