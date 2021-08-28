# %%
import numpy as np
import torch
import torchvision
from utilities import mse, neural_network, relu_derivative

%load_ext autoreload
%autoreload 2

IMAGE_SIZE = 28
OUTPUT_SIZE = 10
NUMBER_HIDDEN_LAYERS = 7
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
outputs, activations = nn.predict(image)
loss = criterion(outputs, label)
# %%

# Let G be the gradient on the unnormalized log probabilities 
# U(2) provided by the cross_entropy operation
g = loss
#gradients[nn.number_hidden_layers] = g
wb_grad = [g]
i = 1



for k in reversed(nn.hidden):
    print(i)
    # Convert the gradient on the layer’s output into a gradient into the pre-
    # nonlinearity activation (element-wise multiplication if f is element-wise)

    #g = np.elementwise(g, relu_derivative(activations[i-1]))

    # Compute gradients on weights and biases 
    act_prev_l = activations[i-1]
    der_nonlin_z = relu_derivative(np.maximum(0, activations[i-1]))
    # WEIGHTS GRADIENT AT EACH LAYER
    # activation of previous layer * 
    # derivative of non-lineariry of z (w*a(l-1)+ b) *
    # derivative of the cost funtion of current activation - gradient of next layer
    weight_gradients = g * np.transpose(activations[i-2])

    # BIAS GRADIENT AT EACH LAYER
    # 1 * 
    # derivative of non-lineariry of z (w*a(l-1)+ b) *
    # derivative of the cost funtion of current activation - gradient of next layer
    bias_gradients = g

    wb_grad.insert(0, np.array([weight_gradients, bias_gradients]))
    #g = g * 
    i += 1


'''
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

'''

# %%
