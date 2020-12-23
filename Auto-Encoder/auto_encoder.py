#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:24:13 2020

@author: mzins
"""

from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np


n_features = 28*28
lr = 0.01
n_epochs = 20
batch_size = 256
test_batch_size = 16
disaply_steps = 1000

n_hidden_1 = 128
n_hidden_2 = 64



train_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True)

test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=test_batch_size,
                                          shuffle=False)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden_1, bias=True)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2, bias=True)

    def forward(self, x):
        f1 = torch.sigmoid(self.fc1(x))
        f2 = torch.sigmoid(self.fc2(f1))
        return f2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_1, bias=True)
        self.fc4 = nn.Linear(n_hidden_1, n_features, bias=True)

    def forward(self, x):
        f3 = torch.sigmoid(self.fc3(x))
        f4 = torch.sigmoid(self.fc4(f3))
        return f4

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        rec = self.decoder(latent)
        return {"rec":rec, "latent":latent}

#%% Training
device = torch.device("cuda")
model = AutoEncoder().to(device)
loss_fn = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

PATH = "trained_weigths.pth"
train = True

if train:
    losses = []
    for e in range(n_epochs):
        for i, (X, _) in enumerate(train_loader):
            flat_X = X.reshape((-1, n_features)).to(device)
            out = model(flat_X)
            optimizer.zero_grad()
            loss = loss_fn(flat_X, out["rec"])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Epoch %d loss: %.4f" % (e, losses[-1]))
    torch.save(model.state_dict(), PATH)
else:
    model.load_state_dict(torch.load(PATH))



#%% Testing

def gridify(img):
    """
        Transform a [m*m, H, W] image into a grid of m x m images.
    """
    m = int(np.sqrt(test_batch_size))
    res = []
    for i in range(m):
        images = []
        for j in range(m):
            images.append(img[i*m+j, :, :])
        res.append(np.hstack(images))
    res = np.vstack(res)
    return res


# with torch.no_grad():
#     for i, (X, _) in enumerate(test_loader):
#         flat_X = X.reshape((-1, n_features)).to(device)
#         out = model(flat_X)
#         rec = out["rec"].cpu().reshape((test_batch_size, 28, 28))

#         gt = gridify(X.squeeze())
#         plt.figure("gt")
#         plt.imshow(gt)
#         rec_viz = gridify(rec)
#         plt.figure("rec")
#         plt.imshow(rec_viz)

#         plt.waitforbuttonpress()
#         plt.close("all")

#%% Analyse

data = {}
for i in range(10):
    data[i] = []
with torch.no_grad():
    for i, (X, Y) in enumerate(train_loader):
        flat_X = X.reshape((-1, n_features)).to(device)
        out = model(flat_X)
        latent = out["latent"].cpu().reshape((-1, n_hidden_2)).numpy()
        for z, x in zip(latent, Y.flatten()):
            data[int(x.item())].append(z)
for i in range(10):
    data[i] = np.vstack(data[i])


# Distance matrix
def dists_to(x, y):
    return np.sqrt(np.sum((x-y)**2, axis=1))
    # return np.divide(y.dot(x) / (np.linalg.norm(x)), np.linalg.norm(y, axis=1))

m = np.zeros((10, 10))
N = 100
for x in range(10):
    print(x)
    for y in range(x, 10):
        tot = 0
        for xx in data[x][:N, :]:
            tot += np.sum(dists_to(xx, data[y][:N, :]))
        avg = tot / (N*N)
        m[x][y] = avg
        m[y][x] = avg


# T-sne
from sklearn.manifold import TSNE

N = 300
dataset = np.vstack([data[i][:N, :] for i in range(10)])
labels = []
for i in range(10):
    labels += [i] * N

tsne = TSNE(n_components=2, verbose=True, n_jobs=16, init="pca")
tsne_results = tsne.fit_transform(dataset)

import seaborn as sns
sns.scatterplot(tsne_results[:, 0], tsne_results[:, 1], palette=sns.color_palette("hls", 10), hue=labels)
