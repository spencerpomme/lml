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
sigmoid = lambda z: 1 / (1 + np.exp(-1 * z))


class MuptilayerPerceptron:
    def __init__(self, layers: list):
        """
        Define the structure of a neural network with a list.
        Params:
            layers: A list of integers, each int is number of nodes in a layer.
                For example:
                layers = [4, 3, 2]
                Then, there would be:
                𝜽1 := 3 x 4 matrix
                𝜽2 := 2 x 3 matrix
                b1 := 3 x 1 matrix (column vector)
                b2 := 2 x 1 matrix (column vector)
                This defines a neural network such that:
                4-feature input layer, 2-class output layerand a hidden layer with 3 nodes.
        Return:
            No return.
        """
        if 0 in layers:
            raise ValueError("Should not have layer with zero node!")
        if len(layers) < 2:
            raise ValueError("Layer size must greater or equal to 2!")
        self.weights, self.biases = self.initialize(layers)


    def initialize(self, layers: list)->(list, list):
        """
        Initialize the neural network. This function realizes behavior
         introduced in __init__ function.
        Params:
            layers: The same as layers param as __init__ method.
        Return:
            A list of np.matrix.
        """
        weights = list(map(np.matrix, list(map(np.ones, list(zip(layers[1:], layers[:-1]))))))
        biases = list(map(np.matrix, list(map(np.ones, list(zip(layers[1:], [1] * (len(layers) - 1)))))))
        return weights, biases

    
    def forwardprop(self, X: np.matrix)->np.matrix:
        """
        Forward propagation phase of neural network.
        Updates the nodes' value of layers.
        Pramms:
            X: A batch of input data samples, a [b_size x feature_num] matrix.
        Return:
            A column vector (as in np.matrix) representing calculated ŷ.
        """
        pass


    def backprop(self):
        """
        Back propagation phase of neural network.
        Updates the weights between layers.
        """
        pass

    
    def train(self, X: np.matrix, y: np.matrix):
        """
        Repeatedly calls forwardprop and backprop function to update the weights
        using gradient descent algorithm.
        """
        pass

    
    def predict(self, sample: np.matrix)->np.matrix:
        """
        Predict output given the input column vector using the trained weights.
        Params:
            sample: Input sample data, a np.matrix object, could be multiple.
        Return:
            result: Output a column vector which is a np.matrix object representing
                    the predicted value given input sample data matrix.
        """
        pass

