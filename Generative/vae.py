#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:12:39 2021

@author: mzins
"""
from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt



n_features = 28*28
lr = 0.004
n_epochs = 30
batch_size = 128
test_batch_size = 16
display_steps = 1000

n_hidden_1 = 128
n_hidden_2 = 64
latent_dim = 2


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
        self.fc_mean = nn.Linear(n_hidden_2, latent_dim)
        self.fc_var = nn.Linear(n_hidden_2, latent_dim)

    def forward(self, x):
        f1 = torch.sigmoid(self.fc1(x))
        f2 = torch.sigmoid(self.fc2(f1))
        means = self.fc_mean(f2)
        vars = F.sigmoid(self.fc_var(f2))
        return {"mean":means, "var": vars}


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, n_hidden_1, bias=True)
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
        latent_distrib = self.encoder(x)
        trick_sampled = torch.cuda.FloatTensor(x.shape[0], latent_dim).normal_()
        z_sampled = trick_sampled * latent_distrib["var"] + latent_distrib["mean"]
        rec = self.decoder(z_sampled)
        return {"rec":rec, "latent_sampled":z_sampled, "latent_distrib":latent_distrib}

    def decode(self, x):
        return self.decoder(x)


def kl_div_normal(mu, log_var):
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1))



#%% Training
device = torch.device("cuda")
model = AutoEncoder().to(device)
loss_fn = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

PATH = "trained_weigths_vae.pth"
train = True


if train:
    losses = []
    for e in range(n_epochs):
        for i, (X, _) in enumerate(train_loader):
            flat_X = X.reshape((-1, n_features)).to(device)
            optimizer.zero_grad()

            out = model(flat_X)
            means = out["latent_distrib"]["mean"]
            vars = out["latent_distrib"]["var"]
            pred_distrib = torch.distributions.normal.Normal(means, vars)

            kl = kl_div_normal(means, torch.log(vars))
            loss = loss_fn(flat_X, out["rec"]) + 0.006 * kl
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print("Epoch %d loss: %.4f" % (e, losses[-1]))
    torch.save(model.state_dict(), PATH)
else:
    model.load_state_dict(torch.load(PATH))


#%%
def gridify(img, size=None):
    """
        Transform a [m*m, H, W] image into a grid of m x m images.
        size: nb images concatenated horizontally and vertically
    """
    if size is None:
        m = int(np.sqrt(img.shape[0]))
    else:
        m = size
    res = []
    for i in range(m):
        images = []
        for j in range(m):
            images.append(img[i*m+j, :, :])
        res.append(np.hstack(images))
    res = np.vstack(res)
    return res

# model.eval()
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
#         if i == 10:
#             break

#%% Analyse

data = {}
for i in range(10):
    data[i] = []
with torch.no_grad():
    for i, (X, Y) in enumerate(train_loader):
        flat_X = X.reshape((-1, n_features)).to(device)
        out = model(flat_X)
        latent = out["latent_distrib"]["mean"].cpu().numpy()
        for z, x in zip(latent, Y.flatten()):
            data[int(x.item())].append(z)
for i in range(10):
    data[i] = np.vstack(data[i])

import seaborn as sns
N = 700
vals = np.vstack(data[i][:N, :] for i in range(10))
labels = []
for i in range(10):
    labels.extend(N*[i])



#%% Display latent space and generate new images
index = 0
def generate_form_z(code):
    model.eval()
    global index
    with torch.no_grad():
        z = code
        z = torch.Tensor(z).unsqueeze(0).cuda()
        gen = model.decode(z)
        img = gen.cpu().numpy().reshape((28, 28))
        plt.figure("generated")
        plt.imshow(img)
        plt.show()
        plt.draw()
        plt.savefig("out_vae/out_%04d.png" % index)
        index += 1


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(ix, iy)
    generate_form_z([ix, iy])

    return ix, iy

fig = plt.figure("latent space")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
sns.scatterplot(vals[:, 0], vals[:, 1], palette=sns.color_palette("hls", 10), hue=labels)

# #%% Decode modified latent code
# import matplotlib.pyplot as plt
# with torch.no_grad():
#     code = data[7][17, :]
#     code[0] *= 2
#     code = data[7][:10, :].sum(axis=0)
#     code = data[7].mean(axis=0)
#     code = torch.Tensor(code).unsqueeze(0)
#     out = model.decode(code.to(device))
#     img = out.cpu().numpy().squeeze().reshape((28, 28))
#     plt.imshow(img)

