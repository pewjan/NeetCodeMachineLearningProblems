import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # matrix multiplcation of X to weights then we take the derivative and subtract it form the current weight 
        #loop through how many iterations to run the gradient descent
        for _ in range(num_iterations):
            #temp np array which we use to update the weights
            tempWeights = np.array(initial_weights)
            for i in range(len(initial_weights)):

                #get the predicted Y value from the weights 
                predicted = self.get_model_prediction(X, initial_weights);

                #gradient descent
                weight = initial_weights[i] - self.learning_rate * self.get_derivative(predicted, Y, len(X), X, i);
                tempWeights[i] = weight;
            initial_weights = tempWeights;
        return np.round(initial_weights, 5);
