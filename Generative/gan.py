#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:45:17 2020

@author: mzins
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

n_features = 28*28

lr_generator = 0.0002
lr_discriminator = 0.0002

lr_gen = 0.0005
lr_disc = 0.0005

batch_size = 128

noise_dim = 20



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
                                          batch_size=batch_size,
                                          shuffle=False)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 7*7*128)
        # self.bn1 = nn.BatchNorm1d(7*7*128)
        self.conv2tr1 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv2tr2 = nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.leaky_relu(x)
        x = x.reshape((-1, 128, 7, 7))
        x = self.conv2tr1(x)
        # x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv2tr2(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        # self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(7*7*128, 512)
        # self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.leaky_relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        # x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# gen_hidden_dim = 256
# disc_hidden_dim = 256

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(noise_dim, gen_hidden_dim)
#         # torch.nn.init.xavier_normal_(self.fc1.weight)
#         self.fc2 = nn.Linear(gen_hidden_dim, n_features)
#         # torch.nn.init.xavier_normal_(self.fc2.weight)
#     def forward(self, x):
#         x = x.flatten(1)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.sigmoid(x)
#         return x.reshape((-1, 1, 28, 28))

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(n_features, disc_hidden_dim)
#         # torch.nn.init.xavier_normal_(self.fc1.weight)
#         self.fc2 = nn.Linear(disc_hidden_dim, 1)
#         # torch.nn.init.xavier_normal_(self.fc2.weight)
#     def forward(self, x):
#         x = x.flatten(1)
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.sigmoid(x)
#         return x

#%% Train
device = torch.device("cuda")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_gen = optim.Adam(generator.parameters(), lr=lr_gen)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr_disc)
train = True
PATH_GEN = "trained_weigths_generator.pth"
PATH_DISC = "trained_weigths_discriminator.pth"


if train:
    n_epochs = 20
    for e in range(n_epochs):
        for i, (X, Y) in enumerate(train_loader):
            noise = torch.rand((X.shape[0], noise_dim), device=device) * 2 - 1

            optimizer_gen.zero_grad()
            gen = generator(noise)
            loss_gen = -torch.mean(torch.log(discriminator(gen)))
            loss_gen.backward()
            optimizer_gen.step()

            optimizer_disc.zero_grad()
            disc_real = discriminator(X.to(device))
            disc_fake = discriminator(gen.detach())
            loss_disc = -torch.mean(torch.log(disc_real) + torch.log(1.0 - disc_fake))
            loss_total = loss_gen + loss_disc
            loss_disc.backward()
            optimizer_disc.step()
        print("epoch %d Loss generator %.4f Loss discriminator %.4f Loss total %.4f" % (e, loss_gen, loss_disc, loss_total))
    torch.save(generator.state_dict(), PATH_GEN)
    torch.save(discriminator.state_dict(), PATH_DISC)
else:
    generator.load_state_dict(torch.load(PATH_GEN))
    discriminator.load_state_dict(torch.load(PATH_DISC))




#%% Generate

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

m = 4
n_tests = 10
generator.eval()
with torch.no_grad():
    for i in range(n_tests):
        noise = torch.rand((m*m, noise_dim), device=device) * 2 - 1
        gen = generator(noise)
        gen = gen.squeeze().cpu().numpy()
        gen_viz = gridify(gen)
        plt.imshow(gen_viz)

        plt.savefig("out_gan/out_%04d.png" % i)
        # plt.waitforbuttonpress()
        # plt.close("all")