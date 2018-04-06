#!/anaconda/envs/tensorflow/bin/python
# -*- coding: utf-8 -*-
"""
Yet another one of the simplest implementation of neural network.
It supports layer structure configuration via a list.
Created on Wed Mar 28 20:07:33 2018

@author: ZHANG PINGCHENG
"""

import numpy as np


# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


class MuptilayerPerceptron:
    """
    This is a class that defines multi-layer perceptron, i.e. a neural network.
    For simplicity, only use sigmoid function as activation function and mini batch
    gradient descent as optimization algorithm.
    Attributes:
        thetas: A list of np.matrix, each represents a weight between two layers of a network.
        biases: A list of np.matrix, each represents a bias to be added to previous layer to 
                calculate the next.
        hidden: Intermediate calculation result of hidden layer matrix, i.e. activations.
    """

    def __init__(self, layers: list, alpha: float, tol: float, batch_size: int):
        """
        Define the structure of a neural network with a list.
        Params:
            layers: A list of integers, each int is number of nodes in a hidden layer.
                For example:
                layers = [4, 3, 2]
                Then, there would be:
                𝜽1 := 3 x 4 matrix
                𝜽2 := 2 x 3 matrix
                b1 := 3 x 1 matrix (column vector)
                b2 := 2 x 1 matrix (column vector)
                This defines a neural network such that:
                4-feature input layer, 2-class output layerand a hidden layer with 3 nodes.
                * Must make sure the definition matches the X and y feature.
            alpha: Learning rate
            tol: tolerance
        Return:
            No return.
        """
        if 0 in layers:
            raise ValueError("Should not have layer with zero node!")
        if len(layers) < 2:
            raise ValueError("Layer size must greater or equal to 2!")
        # Hyperparemeters:
        self.layers = layers
        self.layer_num = len(layers) - 1
        self.alpha = alpha
        self.tol = tol
        self.batch_size = batch_size
        # Weights and gradients:
        self.thetas,\
        self.biases,\
        self.hidden,\
        self.grad_t,\
        self.grad_b,\
        self.grad_h = self._initialize()

    def _initialize(self) -> (list, list, list, list, list, list):
        """
        Initialize the shape of neural network.
        Return:
            A tuple of 6 lists of np.matrix:
            thetas: Weights between layers
            biases: Intercept two 
            hidden: Hidden layer intermediate calculation results. Shaped as:
                    [batch_size x feature_num]
            And initial gradients of them respectively.
        """
        thetas = list(
            map(
                np.matrix,
                list(
                    map(np.ones, list(zip(self.layers[1:], self.layers[:-1])))
                )
            )
        )
        biases = list(
            map(
                np.matrix,
                list(
                    map(
                        np.ones,
                        list(
                            zip(
                                [self.batch_size] * (len(self.layers) - 1),
                                self.layers[1:]
                            )
                        )
                    )
                )
            )
        )
        hidden = list(
            map(
                np.matrix,
                list(
                    map(
                        np.ones,
                        list(
                            zip(
                                [self.batch_size] * (len(self.layers) - 1),
                                self.layers[1:]
                            )
                        )
                    )
                )
            )
        )

        grad_t = list(
            map(
                np.matrix,
                list(
                    map(np.ones, list(zip(self.layers[1:], self.layers[:-1])))
                )
            )
        )
        grad_b = list(
            map(
                np.matrix,
                list(
                    map(
                        np.ones,
                        list(
                            zip(
                                [self.batch_size] * (len(self.layers) - 1),
                                self.layers[1:]
                            )
                        )
                    )
                )
            )
        )
        grad_h = list(
            map(
                np.matrix,
                list(
                    map(
                        np.ones,
                        list(
                            zip(
                                [self.batch_size] * (len(self.layers) - 1),
                                self.layers[1:]
                            )
                        )
                    )
                )
            )
        )
        return thetas, biases, hidden, grad_t, grad_b, grad_h

    def forwardprop(self, X: np.matrix, activation) -> np.matrix:
        """
        Vectorized forward propagation phase of neural network.
        Updates the nodes' value of layers.
        Pramms:
            activation: The choice of ctivation function.
        Return:
            output: A matrix (as in np.matrix) representing n calculated ŷ.
                    The shape of output is [b_size x (number of output layer nodes)]
        """
        self.hidden[0] = activation(X @ self.thetas[0].T + self.biases[0])
        for i in range(1, self.layer_num):
            self.hidden[i] = activation(
                self.hidden[i - 1] @ self.thetas[i].T + self.biases[i]
            )
        output = self.hidden[-1]
        return output

    def backprop(self, X: np.matrix, y: np.matrix):
        """
        Back propagation phase of neural network.
        Updates the weights and biases between layers.
        Params:
            X: A batch of X input data
            y: A batch of corresponding ground truth y
        """
        y_pred = self.forwardprop(X, sigmoid)
        loss = (y - y_pred).sum()
        self.grad_h[-1] = 2 * (y - y_pred)
        self.grad_t[-1] = self.grad_h[-1].T @ self.hidden[-2]
        self.grad_b[-1] = self.grad_h[-1]
        for i in range(self.layer_num - 2, 0, -1):
            self.grad_h[i] = self.grad_h[i + 1] @ self.thetas[i + 1]
            self.grad_t[i + 1] = self.grad_h[i + 1].T @ self.hidden[i + 1]
            self.grad_b[i + 1] = self.grad_h[i + 1]
        return self.grad_h, grad_b, grad_t

    def train(self, X: np.matrix, y: np.matrix, iteration: int):
        """
        Repeatedly calls forwardprop and backprop function to update the weights
        using gradient descent algorithm.
        """
        for i in range(iteration):
            print("Iteration: %s" % i)
            self.backprop(X, y)  # <- bug in backprop function
            for j in range(self.layer_num):
                print(
                    "i: %d/%d j: %d/%d" %
                    (i + 1, iteration, j + 1, self.layer_num)
                )
                self.biases[j] -= self.alpha * self.grad_b[j]
                self.hidden[j] -= self.alpha * self.grad_h[j]
                try:
                    self.thetas[j] -= self.alpha * self.grad_t[j]
                except ValueError:
                    print("self.alpha: {}".format(self.alpha))
                    print("self.grad_t[{}]:\n{}".format(j, self.grad_t[j]))
                    print(
                        "self.grad_t[{}] shape: {}".format(
                            j, self.grad_t[j].shape
                        )
                    )
                    print("self.thetas[{}]:\n{}".format(j, self.thetas[j]))
                    print(
                        "self.grad_t[{}] shape: {}".format(
                            j, self.thetas[j].shape
                        )
                    )

    def predict(self, test_X: np.matrix) -> np.matrix:
        """
        Predict output given the input column vector using the trained weights.
        Params:
            sample: Input sample data, a np.matrix object, could be multiple.
        Return:
            result: Output a column vector which is a np.matrix object representing
                    the predicted value given input sample data matrix.
        """
        return self.forwardprop(test_X, sigmoid)


if __name__ == "__main__":

    mlp = MuptilayerPerceptron([2, 5, 7, 2], 0.1, 0.1, 10)
    X = np.matrix(
        [
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        ]
    ).T
    y = np.matrix([1, 1, 1, 0, 0, 1, 0, 1, 1, 0]).T
    thetas, biases, hidden, grad_t, grad_b, grad_h = mlp._initialize()

    mlp._initialize()

    mlp.train(X, y, 10)
