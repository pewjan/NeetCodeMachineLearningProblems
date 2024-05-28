import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.dataset.mnist.load_data()


class Layer:
    def __init__(self, num_inputs, num_neurons):
                                        # each row is different weight and each column is different neuron
        self.weights = np.random.randn(num_inputs, num_neurons) * .01;
        self.biases = np.random.randn((1, num_neurons));
    def forward(self, input):
        self.inputs = input;
                                        #each feature will be multiplied to each weight then added
        self.output = self.sigmoid(np.dot(self.inputs, self.weights)+self.biases);
        return self.output;
    def sigmoid(self, z):
        return  1/(1+np.exp(-z));
    def sigmoid_der(self, z):
        return z*(1-z);

class NeuralNetwork:
    #layers in the NN
    def __init__(self):
        self.layers = [];
    def addLayer(self, layer):
        self.layers.append(layer);
    #forward propigation and we send each layer the new input by saving the output
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X);
        return X;
        
