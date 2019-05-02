import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([1.0, 2.0, -1.0])

print(step_function(x))

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1, 1, 0])
print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(0, 1)
plt.show()

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 5.1)
plt.show()

def identity_function(x):
    return x