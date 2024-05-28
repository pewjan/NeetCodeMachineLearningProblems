import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.dataset.mnist.load_data()


class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.randn(num_inputs, num_neurons) * .01;
        self.biases = np.random.randn((1, num_neurons));
    def forward(self, input):
        self.inputs = input;
        self.output = self.sigmoid(np.dot(self.inputs, self.weights)+self.biases);
        return self.output;
    def sigmoid(self, z):
        return  1/(1+np.exp(-z));
    def sigmoid_der(self, z):
        return z*(1-z);

class NeuralNetwork:
    pass
