#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:56:27 2023

@author: jason
"""

import matplotlib.pyplot as plt  # for data visualization purposes
import numpy as np
import matplotlib.pyplot as plt  # for data visualization purposes

inputs = [1, 2, 3, 2.5]  # inputs from the previous layer
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]  # the weights for the neuron
bias = [2, 3, 0.5]
# 1st we need to add those values
# The expression is f(w, a , b) = relu*(w0*a0+w1*a1+...+b)


def relu(x):
    return max(0.0, x)

# TODO: check if i the neurons are [n,1] or [1,n]


def activation_function(inputs, weights, num_neurons):

    neurons = np.empty([1, num_neurons])
    for i in range(len(weights)):
        dot_product = np.dot(inputs, weights[i])
        parenthesis = dot_product+bias[i]
        neurons[0, i] = relu(parenthesis)

    return neurons

# print(activation_function(inputs, weights))

# this is one neuron with n inputs
# %%

# lets create 3 neurons (going for a layer)
# the inputs remain the same


neurons_3 = activation_function(inputs, weights, len(weights))
