# %%

import numpy as np
import torch
import torchvision
from utilities import mse, neural_network

%load_ext autoreload
%autoreload 2

IMAGE_SIZE = 28
OUTPUT_SIZE = 10
NUMBER_HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 100
BATCH_SIZE = 10

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                torchvision.transforms.Normalize((0.5,), (1.0,))])

train_set = torchvision.datasets.MNIST(root=".", train=True, transform=trans, download=True)
test_set = torchvision.datasets.MNIST(root=".", train=False, transform=trans, download=True)
# %%
# these loades will output a list of 2 elements, each of them a 
# torch tenson (batch size x picture, batch size x number)
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=BATCH_SIZE,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=BATCH_SIZE,
                shuffle=False)

example = iter(train_loader).next()

# %%
# The backpropagation algorithm works by computing the gradient of the loss function with respect to each weight by the chain rule, computing the gradient one layer at a time, iterating backward from the last layer to avoid redundant calculations of intermediate terms in the chain rule
image = example[0][0][0].numpy() # it's a 6
label = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
criterion = mse
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

nn = neural_network(IMAGE_SIZE**2, OUTPUT_SIZE, NUMBER_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE)
outputs = nn.predict(image)
loss = criterion(outputs, label)
# %%
loss

gradients = {}
g = loss
gradients[len(nn.number_hidden_layers)] = g

for l in reversed(nn.hidden):
    g = g * 





































# %%
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = nn(inputs)
        #outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 
        print('[%d] loss: %.3f' % (i + 1, loss.item()))

    print(f"==== EPOCH LOSS: {running_loss/(i+1)}")



# %%
