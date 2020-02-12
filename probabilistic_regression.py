#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import matplotlib.pyplot as plt


def f(x):
    return np.cos(x) * np.exp(-(0.1*x)**2) + 0.1 * np.sin(2*x)

NB_TRAIN_DATA = 1000
XMIN_TRAIN = -10
XMAX_TRAIN = 10
X_train = np.linspace(XMIN_TRAIN, XMAX_TRAIN, NB_TRAIN_DATA).astype(np.float32)

Y_train = f(X_train)

def train_deterministic(model, loss_fn, X_batches, Y_batches, nb_epochs):
    model.train()
    for e in range(NB_EPOCHS):
        losses = []
        for x, y in zip(X_batches, Y_batches):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("Epoch %d mean loss = %f" % (e, np.mean(losses)))


def test_deterministic(model, X_data, Y_data):
    model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        data = torch.tensor(X_data).unsqueeze(1)
        output = model(data)
    y_pred = output.numpy()
    mean_error = np.sqrt(np.sum((y_pred - Y_data)**2)) / X_test.shape[0]
    plt.scatter(X_test, y_pred, c="red", s=0.1)
    print("Mean error = ", mean_error)



class RegressionNet(nn.Module):
    def __init__(self, layers_size):
        super(RegressionNet, self).__init__()
        self.layers = nn.Sequential()
        layers_size = [1] + layers_size
        for i in range(1, len(layers_size)):
            self.layers.add_module("linear_" + str(i), nn.Linear(layers_size[i-1],
                                                                 layers_size[i]))
            self.layers.add_module("relu_" + str(i), nn.ReLU())
        self.layers.add_module("final", nn.Linear(layers_size[-1], 1))

    def forward(self, x):
        return self.layers(x)




NB_EPOCHS = 500
LEARNING_RATE = 0.0005
GAMMA= 1
BATCH_SIZE = 64

device = torch.device("cpu")

model = RegressionNet([32, 64, 128, 128, 64, 32]).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

X_tensor = torch.Tensor(X_train).unsqueeze(1).to(device)
Y_tensor = torch.Tensor(Y_train).unsqueeze(1).to(device)

X_batches = X_tensor.split(BATCH_SIZE)
Y_batches = Y_tensor.split(BATCH_SIZE)

train_deterministic(model, nn.MSELoss(), X_batches, Y_batches, NB_EPOCHS)


#%% TEST
plt.figure("Training data")
plt.scatter(X_train, Y_train, s=0.1)


ediff = X_train[1] - X_train[0]
X_test = X_train + ediff / 2
X_test = X_test[:-1]
Y_test = f(X_test)

test_deterministic(model, X_test, Y_test)
