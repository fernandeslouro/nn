import numpy as np

INPUT_SIZE = 16
OUTPUR_SIZE = 10
NUMBER_HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 50

class fully_connected_layer(self, previous_size, layer_size):
    def __init__(self):
        self.weights = np.random([previous_size, layer_size])
        self.bias = np.random(layer_size)
        self.output = np.random(layer_size)
    
    def calculate_output(self, previous_output):
        output = previous_output * self.weights + self.bias
        return output

    def backward_pass():

class neural_network(self, input_size, output_size, number_hidden_layers, hidden_layer_size):
    def __init__(self):
        self.input_layer = np.array(input_size)
        self.output_layer = np.array(output_size)
        for i in range(number_hidden_layers):
            self.hidden[i] = fully_connected_layer(hidden_layer_size, hidden_layer_size)

    def train():

    def predict(self, input):
        output = input
        for i in range(number_hidden_layers):
            output *= max(0, self.hidden[i].weights*output+bias)



nn = neural_network(INPUT_SIZE, OUTPUT_SIZE, NUMBER_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE)

prediction = nn.predict(data)
