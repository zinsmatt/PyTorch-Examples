import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from distributions import SmoothOneHot, Dirichlet

BATCH_SIZE = 20
BATCH_SIZE_TEST = 20
LEARNING_RATE = 0.001
GAMMA = 0.1
NB_EPOCHS = 10


def train(model, optimizer, loss_fn, data_loader, nb_epochs):
    model.train()
    loss_fn.train()
    
    for e in range(nb_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            # loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 500 == 0:
                print("Train epoch %d batch %d =>> Loss = %f" % (e, batch_idx, loss))
                

def test(model, data_loader):
    model.eval()
    nb_correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            
            data, target_gpu = data.to(device), target.to(device)
            output = model(data)["mean"]
            loss = F.nll_loss(output, target_gpu)
            pred = output.argmax(dim=1, keepdim=True).cpu().numpy().flatten()
            
            correct = pred == target.numpy()
            print("%d correct / %d" % (correct.sum(), target.shape[0]))
            nb_correct += correct.sum()
            total += target.shape[0]
            
    print("TOTAL: %d correct / %d" % (nb_correct, total))
    print("Accuracy: ", float(nb_correct) / total)
    



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data/", train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=BATCH_SIZE, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data/", train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=BATCH_SIZE_TEST, shuffle=True)



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    

class LeNetProbOut(nn.Module):
    def __init__(self):
        super(LeNetProbOut, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 20)
        
    def forward(self, x):
        # x = self.conv1(x)
        # print(x)
        # x = F.relu(x)
        # print(x)
        # x = F.max_pool2d(x, 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # print(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        mean, log_variance = x.chunk(chunks=2, dim=1)
        # print("mean = ", mean)
        # print("log_variance = ", log_variance)
        return {"mean": mean, "variance": torch.exp(log_variance)}
        
class DirichletProbOutLoss(nn.Module):
    def __init__(self):
        super(DirichletProbOutLoss, self).__init__()
        
        self._log_bias_c1 = nn.Parameter(-torch.ones(1, 1) * 0.0, requires_grad=True)
        self._log_bias_c2 = nn.Parameter(-torch.ones(1, 1) * 0.0, requires_grad=True)
        self._softmax = nn.Softmax(dim=1)
        self._smoothed_onehot = SmoothOneHot(1e-3, True)

        self._dirichlet = Dirichlet(argmax_smoothing=0.5)
        
    def forward(self, output, target):
        output_mean = output["mean"]
        output_var = output["variance"]
        
        num_classes = 10
        smoothed_onehot = Variable(self._smoothed_onehot(target.data, num_classes=num_classes),
                                   requires_grad=False)
        c1 = F.softplus(self._log_bias_c1)
        c2 = F.softplus(self._log_bias_c2)
        
        mu = self._softmax(output_mean)
        stddev = torch.sqrt(torch.sum(mu**2 * output_var, dim=1,keepdim=True))
        print("c1 = ", c1)
        print("c2 = ", c2)
        s = 1.0 / (1.0e-4 + c1 + c2 * stddev)
        alpha = mu * s
           
        total_loss = -self._dirichlet(alpha, smoothed_onehot).mean()
        # print("total_loss = ", total_loss)
        return total_loss

device = torch.device("cuda")

# model = LeNet().to(device)
model = LeNetProbOut().to(device)



optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

# train(model, optimizer, F.nll_loss, train_loader, 2)
train(model, optimizer, DirichletProbOutLoss().to(device), train_loader, 10)

# torch.save(model.state_dict(), "model_trained.pth")
# model.load_state_dict(torch.load("model_trained.pth"))

test(model, test_loader)
            

        
#%%
import matplotlib.pyplot as plt
import scipy.special

plt.close("all")

img = plt.imread("custom_images/test.png")
img = (img - 0.1307) / 0.3081
test = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
model.eval()
with torch.no_grad():
    res = model(test.to(device))

# distrib = np.exp(res["mean"].cpu().numpy().flatten())
distrib = res["mean"].cpu().numpy().flatten()
variances = res["variance"].cpu().numpy().flatten()

m = scipy.special.softmax(distrib)
scale = np.sqrt(np.sum(np.multiply(m**2, variances)))
print(scale)

plt.figure("image")
plt.imshow(img)
plt.figure("distrib")
plt.bar(np.linspace(0, 9, 10), m)

plt.figure("variances")
plt.bar(np.linspace(0, 9, 10), variances)


#%%

model.eval()
with torch.no_grad():
    for data, target in test_loader:        
        data, target_gpu = data.to(device), target.to(device)
        output = model(data)
        
        distrib = output["mean"].cpu().numpy().flatten()
        variances = output["variance"].cpu().numpy().flatten()
        
        m = scipy.special.softmax(distrib)
        scale = np.sqrt(np.sum(np.multiply(m**2, variances)))
        print(scale)
