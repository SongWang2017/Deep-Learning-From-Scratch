import numpy as np
from nn.mnist import load_mnist
from nn.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

train_loss_list = []
train_acc_list = []
test_acc_list = []


#hyperparameters
iters_num = 100000
train_size = x_train.shape[0]
batch_size = 32
iter_per_epoch = max(train_size / batch_size, 1)

learning_rate = 0.00001

network = TwoLayerNet(input_size=784, hiddent_size=50, output_size=10)
#print(network.params)


for i in range(iters_num):
    #get mini_batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #calculate gradient
    #grad = network.numerical_gradient(x_batch, t_batch)
    #bp gradient #
    grad = network.gradient(x_batch, t_batch)

    #update parameters
    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]

    #logging the learning process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #logging the accuracy every batch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))


plt.figure(figsize=(16, 6))
plt.plot(train_loss_list)
plt.show()

