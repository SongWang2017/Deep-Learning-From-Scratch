import sys, os
sys.path.append(os.pardir)
import numpy as np
from nn.mnist import load_mnist
from nn.two_layer_net import TwoLayerNet

#read data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hiddent_size=50, output_size=10)

x_batch = x_train[:5]
t_batch = t_train[:5]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_numerical[key] - grad_backprop[key]))
    print(diff)
