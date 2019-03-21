# GenNet
An evolutionary algorithm approach to design the optimal MLP architecture in PyTorch for classifying MNIST images.

## Use
Enter "python main.py" in command line to see performance of the "gennet" model.

## Prerequisites
Python \
PyTorch \
matplotlib

## Code Details
"data.py" contains the torch calls to retrieve images form the MNIST database. \
"nn.py" contains the outline for the MLP neural network. \
"evol.py" contains the evolutionary algorithm used to discover new architectures and finds the best one.

## Results
The evolutionary algorithm found the following architecture to be optimal after 10 generations with a population size of 20 \
NN( \
  (layers): Sequential( \
    (0): Linear(in_features=784, out_features=421, bias=True) \
    (1): CELU(alpha=1.0) \
    (2): Linear(in_features=421, out_features=10, bias=True) \
    (3): CELU(alpha=1.0) \
  ) \
) \
This architecture was able to achieve 96.6% accuracy on new images.
