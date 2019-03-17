import torch
import torchvision
import torchvision.datasets as datasets

def get_mnist_train():
	return datasets.MNIST(root = './data', train = True, download = True, transform = None)

def get_mnist_test():
	return datasets.MNIST(root = './data', train = False, download = True, transform = None)