import numpy as np

def cross_entropy_loss(batch_truth, batch_predictions):
    return -1/len(batch_truth) * np.inner((batch_truth, np.log(batch_predictions))


class fully_connected_layer(self, previous_size, layer_size):
    def __init__(self):
        self.weights = np.random([previous_size, layer_size])
        self.bias = np.random(layer_size)
        self.output_layer = np.random(layer_size)
    
    def calculate_output(self, previous_output):
        output = previous_output * self.weights + self.bias
        return output

    def backpropagate_batch(self, batch_data):
        for entry in batch_data:
            batch_output.append(self.calculate_output(entry))
        batch_error = cross_entropy_loss(batch_data.labels, batch_output)

class neural_network(self, input_size, output_size, number_hidden_layers, hidden_layer_size):
    def __init__(self):
        self.input_layer = np.array(input_size)
        self.output_layer = np.array(output_size)
        for i in range(number_hidden_layers):
            self.hidden[i] = fully_connected_layer(hidden_layer_size, hidden_layer_size)

    #def train(data):
    #    predictions = nn.predict(data)
    #    loss = cross_entropy_loss(predictions, data.labels)
        

    def predict(self, input_values):
        prediction = input_values
        for i in range(number_hidden_layers):
            prediction *= max(0, self.hidden[i].weights*prediction+bias)
        prediction *= max(0, self.output_layer.weights*prediction+bias)
        prediction = softmax(prediction)

