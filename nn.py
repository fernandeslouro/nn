import numpy as np

INPUT_SIZE = 16
OUTPUR_SIZE = 10
NUMBER_HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 50

class fully_connected_layer(self, previous_size, layer_size):
    def __init__(self):
        self.weights = np.random(previous_size * layer_size)
        self.bias = np.random(layer_size)
        self.output = np.random(layer_size)
        

class neural_network(self, INPUT_SIZE, output_size, NUMBER_HIDDEN_LAYERS, HIDDEN_LAYER_SIZE):
    def __init__(self):
        self.input_layer = np.array(IMPUT_SIZE)
        self.output_layer = np.array(OUTPUT_SIZE)
        
    def train():




input_layer = np.array(IMPUT_SIZE)

for i in range(NUMBER_HIDDEN_LAYERS):
    lay = fully_connected_layer(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)

output_layer = np.array(OUTPUT_SIZE)

