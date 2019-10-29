#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import torch
import torch.nn as nn





pts = np.random.random((3, 100))
pts[2, :] += 3

K = np.array([[250, 0, 300],
              [0, 255, 250],
              [0.0, 0.0, 1.0]])

uvs = K @ pts
uvs_n = uvs / uvs[2, :]



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lin = nn.Linear(3, 3, bias=False)
        
    def forward(self, x):
        x_n = x / x[:, 2].view((-1, 1))
        return self.lin(x_n)
    
    
    
def MyLoss(x, y):
#    print("x = ", x)
#    print("y = ", y)
    l = nn.MSELoss()
#    x_n = x / x[:, 2].view((-1, 1))
#    print(x_n)
#    print(y)
    l1 = nn.MSELoss()
    return l(x[:, :2], y[:, :2]) + l1(network.lin.weight[2, :], torch.tensor([0.0, 0.0, 1.0])) 


network = Network()
optimizer = torch.optim.Adam(network.parameters(), lr=0.1, weight_decay=0.0001)

loss_fn = MyLoss

nn.init.uniform_(network.lin.weight)

network.lin.weight.data[2, 2] = 1.0
network.lin.weight.data[1, 0] = 0.0
network.lin.weight.data[2, 0] = 0.0
network.lin.weight.data[2, 1] = 0.0

temp_results = []

for i in range(20000):
    x = torch.Tensor(pts.T)
    y = torch.Tensor(uvs_n.T)
    
    output = network(x)
    
    loss = loss_fn(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss.item())
    if i % 50 == 0:
        temp_results.append(network.lin.weight.detach().numpy().copy())

    
K_est = network.lin.weight.detach().numpy()
#K_est /= K_est[2, 2]

temp_results.append(K_est)

#%%
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


for i in range(len(temp_results)):
    K_est = temp_results[i]
    outputs = K_est @ pts
    outputs /= outputs[2, :]
    plt.scatter(uvs_n[0, :], uvs_n[1, :], c='r')
    plt.scatter(outputs[0, :], outputs[1, :], marker='+', c='b')
    plt.savefig("output/result_optim_%05d.png" % i)
    plt.close()
