import sys, os

sys.path.append(os.pardir)
import numpy as np
from nn.output import softmax
from nn.mnist import load_mnist
from nn.activation import relu, sigmoid
from learning.grad import numerical_gradient
from learning.loss import cross_entropy_error

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


'''
net = simpleNet()
print(net.W)


x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

t = np.array([0, 0, 1])
print(net.loss(x, t))
'''
net = simpleNet()
def f(W):
    return net.loss(x, t)

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
dW = numerical_gradient(f, net.W)

print(dW)



