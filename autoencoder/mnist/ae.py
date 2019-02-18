import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


class AE(nn.Module):

    def __init__(self, input_size):
        super(AE, self).__init__()
        self.l1 = nn.Linear(input_size, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, 2)
        self.l4 = nn.Linear(2, 50)
        self.l5 = nn.Linear(50, 50)
        self.out = nn.Linear(50, input_size)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.l1(x))
        x = self.tanh(self.l2(x))
        x = self.l3(x)
        x = self.tanh(self.l4(x))
        x = self.tanh(self.l5(x))
        return self.out(x)

# define normalize transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download / create MNIST dataset (also works for FashionMNIST)
train_set = torchvision.datasets.MNIST('./autoencoder/mnist/data', download=True, transform=transform)
test_set = torchvision.datasets.MNIST('./autoencoder/mnist/data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=25, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=25)

# define input_size
input_size = train_set.train_data[0].shape[0] * train_set.train_data[0].shape[1] # can't think of a smarter version atm

# hyperparameter
lr = 1e-4

# init Autoencoder
ae = AE(input_size)
optimizer = optim.Adam(ae.parameters(),lr=lr)
criterion = nn.MSELoss()

# helper class to de-normalize images
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

for epoch in range(100):

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        # get data batch
        inputs, labels = data

        # flatten the input images (batch_size, input_size)
        inputs = inputs.view(inputs.size(0),-1)

        # reset gradients
        optimizer.zero_grad()

        # perform forward step
        outputs = ae(inputs)

        # calculate loss + gradients
        loss = criterion(outputs, inputs)
        loss.backward()

        # update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 600 == 599:    # print every 500 mini-batches
            print(f'[Epoch: {epoch + 1}, Batch: {(i+1)}] loss: {running_loss / 600:.3f}')
            running_loss = 0.0

        # save images from last batch
        if i == 2399:
            # reshape to original dimensions + de-normalize
            outputs = to_img(outputs.view(25, 28, 28))
            inputs = to_img(inputs.view(25, 28, 28))
            torchvision.utils.save_image(torchvision.utils.make_grid(outputs,nrow=5), f'./autoencoder/mnist/pred/epoch_{epoch+1}_pred.png')
            torchvision.utils.save_image(torchvision.utils.make_grid(inputs, nrow=5), f'./autoencoder/mnist/pred/epoch_{epoch+1}_target.png')