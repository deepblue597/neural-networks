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


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # one hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

# to prevent overflow we (values too big to calculate) what we do is we subtract the max value from the inputs
# now the values are between zero and 1.
# then we divide by the sum so we have our probabilites


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
layer_2 = Layer_Dense(3, 3)

activation_1 = Activation_ReLU()
activation_2 = Activation_Softmax()

layer_1.forward(X)
# print(layer_1.output)

activation_1.forward(layer_1.output)
layer_2.forward(activation_1.output)

activation_2.forward(layer_2.output)
print(activation_2.output[:5])

# =============================================================================
# Mean Squared Error (MSE):
#
# Used for regression problems where the target variable is a continuous value.
# The loss is calculated as the square of the difference between the predicted value (a) and the actual target value (y): (a - y)^2.
# MSE is often used when the output of the MLP represents a single continuous value, and you want to measure the difference between the predicted and actual values.
#
# Categorical Cross-Entropy:
#
# Used for classification problems where the target variable is a categorical variable (e.g., one-hot encoded class labels).
# The loss is calculated based on the predicted class probabilities (a) and the actual class labels (y). It measures the dissimilarity between the predicted and actual class distributions using the cross-entropy formula.
# Categorical cross-entropy is commonly used when the output of the MLP represents class probabilities, and you want to measure the difference between the predicted class distribution and the true class distribution.
# =============================================================================
# log in python is ln

# %%


# print(np.log(np.e))
# # y = np.array([[1, 0 , 0 ],
# #      [0, 1 , 0],
# #      [0, 0 , 1]])

# y_pred = np.array([[0.66 , 0.33 , 0.01],
#           [0.9 , 0.05 , 0.05],
#           [0.2 , 0.2 , 0.6]])

# class_target =[ 0 , 1 , 2]
# loss = - np.sum(y * np.log(y_pred), axis=1 , keepdims=True)


# print(y_pred[range(len(y_pred)),class_target])

# y_new = y_pred[range(len(y_pred)),class_target]
# loss_new = - np.log(y_new) #this is correct. we take the correct classifications with the predictions and we create the values

# if a prediction is completely wron (pred = 0 ) then the lg is inf
# to prevent that we use clip
# print(1e-7)
# loss_new = np.clip(loss_new, 1e-7, 1 - 1e-7)

# %%

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation_2.output, y)
print("Loss:", loss)
