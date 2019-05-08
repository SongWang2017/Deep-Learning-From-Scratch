import sys, os

sys.path.append(os.pardir)
from nn.mnist import load_mnist
import numpy as np
import pickle
from PIL import Image
from activation import sigmoid
from output import softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = True, one_hot_label = False)

    return x_test, t_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y



x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)

    if p == t[i]:
        accuracy_cnt += 1
print("accuracy:" + str(float(accuracy_cnt) / len(x)))

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

#normalize whether to normalize the images into 0-1 if false the value will be between 0-255
#flatten whether to flatten the images into arrays of 1d, if false the images are(1, 28, 28)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)


batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i : i + batch_size])

print("accuracy:" + str(float(accuracy_cnt) / len(x)))


print(x_train.shape)
print(t_train.shape)

print(x_test.shape)
print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)