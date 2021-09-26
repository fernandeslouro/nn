import numpy as np

# square error is usually a poor choice for a classification task such as MNIST
# this is the cross entropy loss


def cross_entropy_loss(truth, predictions):
    # return -1/len(batch_truth) * np.inner((batch_truth,
    # np.log(batch_predictions)))
    return -sum(truth*np.log(predictions))


def mse(pred, label):
    return np.square(pred-label).mean()


def mse_derivative(pred, correct):
    return 2 * (pred - correct)


def softmax(inputs):
    return np.exp(inputs)/np.sum(np.exp(inputs))


def relu_derivative(arr):
    return (arr > 0) * 1


class fully_connected_layer():
    def __init__(self, previous_size, layer_size):
        self.previous_size = previous_size
        self.layer_size = layer_size
        self.weights = np.random.rand(self.previous_size, self.layer_size)
        # self.bias = np.expand_dims(np.random.rand(self.layer_size), axis=1)
        self.bias = np.ones([layer_size, 1])*0.1
        self.output_layer = np.random.rand(self.layer_size)

    def calculate_output(self, previous_output):
        mult = np.matmul(self.weights.transpose(), previous_output)
        summation = mult + self.bias
        output = np.maximum(0, summation)
        return output, summation

    def backpropagate_batch(self, batch_data):
        batch_output = []
        for entry in batch_data:
            batch_output.append(self.calculate_output(entry))
        batch_error = cross_entropy_loss(batch_data.labels, batch_output)
        return batch_error


class neural_network():
    def __init__(
            self, input_size, output_size,
            number_hidden_layers, hidden_layer_size):

        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = np.array(input_size)
        self.number_hidden_layers = np.array(number_hidden_layers)
        self.hidden_layer_size = np.array(hidden_layer_size)

        self.hidden = []
        for i in range(self.number_hidden_layers):
            if i == 0:
                self.hidden.append(fully_connected_layer(
                    self.input_size,
                    self.hidden_layer_size))
            if i == self.number_hidden_layers-1:
                self.hidden.append(fully_connected_layer(
                    self.hidden_layer_size,
                    self.output_size))
            else:
                self.hidden.append(fully_connected_layer(
                    self.hidden_layer_size,
                    self.hidden_layer_size))

    def predict(self, input_values):
        prediction = np.expand_dims(input_values.flatten(), axis=1)
        activations = []
        for hidden_layer in self.hidden:
            layer_output, activation = hidden_layer.calculate_output(prediction)
            activations.append(np.array([layer_output, activation]))
        prediction = softmax(layer_output)
        return prediction, activations

