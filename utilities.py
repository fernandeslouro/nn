import numpy as np

 
def cross_entropy_loss(batch_truth, batch_predictions):
    return -1/len(batch_truth) * np.inner((batch_truth, np.log(batch_predictions)))  

def softmax(inputs):
    print(inputs)
    return np.exp(inputs)/np.sum(np.exp(inputs))




class fully_connected_layer():
    def __init__(self, previous_size, layer_size):
        self.previous_size = previous_size
        self.layer_size = layer_size
        self.weights = np.random.rand(self.previous_size, self.layer_size)
        self.bias = np.expand_dims(np.random.rand(self.layer_size), axis=1)
        self.output_layer = np.random.rand(self.layer_size)
    
    def calculate_output(self, previous_output):
        print(self.weights.transpose().shape, previous_output.shape)
        mult = np.matmul(self.weights.transpose(), previous_output)
        print(mult.shape, self.bias.shape)
        summation = mult + self.bias
        output = np.maximum(0, summation)
        print(f"OUTPUT - {output.shape}")
        return output

    def backpropagate_batch(self, batch_data):
        batch_output = []
        for entry in batch_data:
            batch_output.append(self.calculate_output(entry))
        batch_error = cross_entropy_loss(batch_data.labels, batch_output)
        return batch_error

class neural_network():
    def __init__(self, input_size, output_size, number_hidden_layers, hidden_layer_size):
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = np.array(input_size)
        self.number_hidden_layers = np.array(number_hidden_layers)
        self.hidden_layer_size = np.array(hidden_layer_size)

        self.hidden = [] 
        for i in range(self.number_hidden_layers):
            if i == 0:
                self.hidden.append(fully_connected_layer(self.input_size, self.hidden_layer_size))
            if i == self.number_hidden_layers-1:
                self.hidden.append(fully_connected_layer(self.hidden_layer_size, self.output_size))
            else:
                self.hidden.append(fully_connected_layer(self.hidden_layer_size, self.hidden_layer_size))

    def predict(self, input_values):
        prediction = np.expand_dims(input_values.flatten(), axis=1)
        for i, hidden_layer in enumerate(self.hidden):
            prediction = hidden_layer.calculate_output(prediction)
            print(f"LAYER {i+1}")
        prediction = softmax(prediction)
        return prediction
