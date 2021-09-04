# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class1 = 2
class2 = 0

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        #raise NotImplementedError("You need to write this part!")
        # based on the code on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
        self.lrate = lrate
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.optimizer = optim.SGD(self.parameters(), self.lrate, momentum=0.9)

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #raise NotImplementedError("You need to write this part!")

        #standard_data = (x - float(x.mean())) / float(x.std())
        # based on the code on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
        x = (x - x.mean()) / x.std()
        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        #raise NotImplementedError("You need to write this part!")
        self.optimizer.zero_grad()
        y_pred = self.forward(x)
        L = self.loss_fn(y_pred, y)
        L.backward()
        self.optimizer.step()
        return L.item()
        #return 0.0

def fit(train_set,train_labels,dev_set,n_iter,batch_size=50):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    #raise NotImplementedError("You need to write this part!")
    losses = []
    yhats = []
    lrate = 0.04
    loss_fn = nn.CrossEntropyLoss()
    in_size = len(train_set[0])
    out_size = 2
    net = NeuralNet(lrate, loss_fn, in_size, out_size)
    # Data Standardization move in forward function
    #standard_data = (train_set - train_set.mean()) / train_set.std()
    #standard_data_dev = (dev_set - dev_set.mean()) / dev_set.std()
    for i in range(n_iter):
        n = len(train_set) // batch_size
        if i < (n):
            batch_data = train_set[i * batch_size : (i+1) * batch_size]
            batch_labels = train_labels[i * batch_size : (i+1) * batch_size]
        # if i exceed the size the train dataset
        else:
            batch_data = train_set[(i-n) * batch_size : (i-n+1) * batch_size]
            batch_labels = train_labels[(i-n) * batch_size : (i-n+1) * batch_size]
        if(len(batch_data) == 0):
            continue
        loss = net.step(batch_data, batch_labels)
        losses.append(loss)
    # compute yhats for dev_set
    temp = net(dev_set).detach().numpy()
    for i in range(len(temp)):
        yhats.append(np.argmax(temp[i]))
    return losses, yhats, net