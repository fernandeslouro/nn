# nn

This project is meant to increase my knowledge of neural networks at a lower level.
It's a simple implementation of a fully connected neural network, to be trained on MNIST.

Implementation is in progress.

Current limitations:
 - The netwok can't be trained since backprop is being implemented :^)
 - No minibatch support (during training, one data instance will lead to one gradient descent)
 - Only architecture change allowed is the number and width of hidden layers
 - Loss functions and non-linearities are hard-set in the code
