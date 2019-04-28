import torch
import torchvision
import torchvision.datasets as datasets

def get_mnist_train():
	return datasets.MNIST(root = './data', train = True, transform = torchvision.transforms.ToTensor(), download = True)

def get_mnist_test():
	return datasets.MNIST(root = './data', train = False, transform = torchvision.transforms.ToTensor(), download = True)