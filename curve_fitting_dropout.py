#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class CurveFittingNet(nn.Module):
    def __init__(self, h1, h2):
        super(CurveFittingNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(1, h1),
                nn.ReLU())
        self.layer2 = nn.Sequential(
                nn.Linear(h1, h2),
                nn.ReLU())
        self.fc = nn.Linear(h2, 1)
        
    def forward(self, x):
        print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        return self.fc(out)
    
    
    
#%% Data preparation
        
X = np.linspace(-5.0, 5.0, 10)
Y = 0.25 * np.cos(X) * X

plt.scatter(X, Y)
plt.xlim([-10.0, 10.0])
plt.ylim([-2.0, 2.0])


#%% Training

model = CurveFittingNet(10, 10)
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_epochs = 1000

for e in range(n_epochs):
    inputs = torch.Tensor(X).unsqueeze(1)
    labels = torch.Tensor(Y).unsqueeze(1)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("epoch %d loss %f" % (e, loss))
    
#%% Eval
model.eval()
X_eval = np.linspace(-5.0, 5.0, 100)

with torch.no_grad():
    inputs_eval = torch.Tensor(X_eval).unsqueeze(1)
    outputs_eval = model(inputs_eval)
    res = outputs_eval.numpy().ravel()
    
    plt.scatter(X_eval, res, c="red", s=0.1)
        