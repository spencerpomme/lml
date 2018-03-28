#!/anaconda/envs/tensorflow/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:07:33 2018

@author: ZHANG PINGCHENG
"""

import numpy as np


class MuptilayerPerceptron:
    def __init__(self, netstruct: list):
        self.netstruct = self.initialize(netstruct)
    

    def forwardprop(self):
        pass


    def backprop(self):
        pass

    
    def initialize(self, netstruct: list)->list:
        pass
    

    def train(self, X: np.matrix, y: np.matrix):
        pass

    
    def predict(self, sample: np.matrix)->np.matrix:
        pass

