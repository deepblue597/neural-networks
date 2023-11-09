#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:56:27 2023

@author: jason
"""

import matplotlib.pyplot as plt  # for data visualization purposes
import numpy as np
import matplotlib.pyplot as plt  # for data visualization purposes
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# X = [[1 , 2 , 3, 2.5 ],
#           [2.0 , 5.0 , -1.0 , 2.0],
#           [-1.5 , 2.7 , 3.3 , -0.8 ]
# ] # inputs from the previous layer
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]  # the weights for the neuron
bias = [2, 3, 0.5]

weights_2 = [[0.1, - 0.14, 0.5,],
             [-0.5, 0.12, -0.33],
             [-0.44, 0.73, -0.13]]  # the weights for the neuron
bias_2 = [-1, 2, -0.5]
# 1st we need to add those values
# The expression is f(w, a , b) = relu*(w0*a0+w1*a1+...+b)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0.0, inputs)

# this is used to make out the probabilites for the last part.
# if we use the Relu and the ouput is negative then we will have 0 on one hand and 100 on the other.
# even wrose if both are negative then we wont have a definite answer so we wont be able to find out the probabilites


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites


# to prevent overflow we (values too big to calculate) what we do is we subtract the max value from the inputs
# now the values are between zero and 1.
# then we divide by the sum so we have our probabilites


# TODO: check if i the neurons are [n,1] or [1,n]
# def layer_function(inputs , weights , bias):

#     layer = [] #initialize the neurons
#     parenthesis = np.dot(inputs, np.transpose(weights)) + bias #the parenthesis of the function. Question: why dont we do weight * inputs ?
#     #look S. Haykin, Neural Networks pg 142
#     layer = relu(parenthesis) #the value of neurons

#     return layer

# print(activation_function(inputs, weights))

# this is one neuron with n inputs
# %%

# lets create 3 neurons (going for a layer)
# the inputs remain the same


# layer_1 = layer_function(X, weights , bias)
# layer_2 = layer_function(layer_1, weights_2 , bias_2)

# shape
# [1, 2 ,3 ,4] --> shape(4)
# [[1 , 2 , 3 ,4 ],
#   [3 , 4  , 5 , 6]] --> shape(2,4)
# tensor is an object that can be represented as an array

# %%

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# say we have 4 inputs (a sample of 4) and we want 3 neurons. To create the appropriate weights we need
# to create a 4x3 matrix. So we create it by the randn
# print(np.random.normal(loc=0.0, scale=1.0, size=(4,3)))


layer_1 = Layer_Dense(2, 3)  # 2 bacause we have x y attributes no more
layer_2 = Layer_Dense(3, 2)

activation_1 = Activation_ReLU()
activation_2 = Activation_Softmax()

layer_1.forward(X)
# print(layer_1.output)

activation_1.forward(layer_1.output)
layer_2.forward(activation_1.output)

activation_2.forward(layer_2.output)
print(layer_2.output)
