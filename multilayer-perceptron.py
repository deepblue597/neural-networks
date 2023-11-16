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
from nnfs.datasets import vertical_data
import nnfs
from timeit import timeit
from zipfile import ZipFile
import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
from nnfs.datasets import sine_data
from sys import exit


# %%
# URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
# FILE = 'fashion_mnist_images.zip'
# FOLDER = 'fashion_mnist_images'


# if not os.path.isfile(FILE):
#     print(f'Downloading {URL} and saving as {FILE}...')
#     urllib.request.urlretrieve(URL, FILE)


# print('Unzipping images...')
# with ZipFile(FILE) as zip_images:
#     zip_images.extractall(FOLDER)


# print('Done!')
# #%%
# labels = os.listdir('fashion_mnist_images/train')
# print(labels)
# files = os.listdir('fashion_mnist_images/train/0')
# print(files[:10])
# print(len(files))


# image_data = cv2.imread('fashion_mnist_images/train/7/0002.png',
#                         cv2.IMREAD_UNCHANGED)
# # print(image_data)


# # np.set_printoptions(linewidth=200)

# plt.imshow(image_data, cmap='gray')
# plt.show()

# # Loads a MNIST dataset
# def load_mnist_dataset(dataset, path):

#     # Scan all the directories and create a list of labels
#     labels = os.listdir(os.path.join(path, dataset))

#     # Create lists for samples and labels
#     X = []
#     y = []

#     # For each label folder
#     for label in labels:
#         # And for each image in given folder
#         for file in os.listdir(os.path.join(path, dataset, label)):
#             # Read the image
#             image = cv2.imread(os.path.join(
#                         path, dataset, label, file
#                     ), cv2.IMREAD_UNCHANGED)

#             # And append it and a label to the lists
#             X.append(image)
#             y.append(label)

#     # Convert the data to proper numpy arrays and return
#     return np.array(X), np.array(y).astype('uint8')

# def create_data_mnist(path):

#     # Load both sets separately
#     X, y = load_mnist_dataset('train', path)
#     X_test, y_test = load_mnist_dataset('test', path)

#     # And return all the data
#     return X, y, X_test, y_test


# X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# X = (X.astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# # Reshape to vectors
# X = X.reshape(X.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

# %%

# print(y[6000:6010])

# keys = np.array(range(X.shape[0]))
# print(keys[:10])
# np.random.shuffle(keys)
# print(keys[:10])
# X = X[keys]
# y = y[keys]
# plt.imshow((X[8].reshape(28, 28)))  # Reshape as image is a vector already
# plt.show()
# print(y[8])

# BATCH_SIZE = 128
# EPOCHS = 10


# #to make batches
# steps = X.shape[0] // BATCH_SIZE

# if X.shape[0] % BATCH_SIZE != 0:
# steps += 1

# for epoch in range(EPOCHS):
#     for step in range(steps):
#         batch_X = X[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
#         batch_y = y[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
# %%
nnfs.init()
# X, y = vertical_data(samples=100, classes=3)
X_train, y_train = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)


# # X = [[1 , 2 , 3, 2.5 ],
# #           [2.0 , 5.0 , -1.0 , 2.0],
# #           [-1.5 , 2.7 , 3.3 , -0.8 ]
#           # ] # inputs from the previous layer
# weights = [[0.2 , 0.8 , -0.5, 1.0],
#            [0.5 , -0.91 , 0.26 , -0.5],
#            [-0.26, -0.27 , 0.17 , 0.87] ] # the weights for the neuron
# bias = [2, 3 , 0.5]

# weights_2 = [[0.1 ,- 0.14 , 0.5,],
#            [-0.5 , 0.12 , -0.33],
#            [-0.44, 0.73 , -0.13] ] # the weights for the neuron
# bias_2 = [-1, 2 , -0.5]
# 1st we need to add those values
# The expression is f(w, a , b) = relu*(w0*a0+w1*a1+...+b)

class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):

        self.inputs = inputs

        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    # this needs to be done because different activations
    # use different ways to calculate predictions and we
    # want to make the proccess easier
    # this is used to make out the probabilites for the last part.
    # if we use the Relu and the ouput is negative then we will have 0 on one hand and 100 on the other.
    # even wrose if both are negative then we wont have a definite answer so we wont be able to find out the probabilites

    def predictions(self, outputs):
        return outputs


# class Activation_ReLU:

#     # Forward pass
#     def forward(self, inputs):
#       # Remember input values
#         self.inputs = inputs
#         # Calculate output values from inputs
#         self.output = np.maximum(0, inputs)

#     # Backward pass
#     def backward(self, dvalues):
#         # Since we need to modify original variable,
#         # let's make a copy of values first
#         self.dinputs = dvalues.copy()

#         # Zero gradient where input values were negative
#         self.dinputs[self.inputs <= 0] = 0
# this is used to make out the probabilites for the last part.
# if we use the Relu and the ouput is negative then we will have 0 on one hand and 100 on the other.
# even wrose if both are negative then we wont have a definite answer so we wont be able to find out the probabilites


class Activation_Softmax:
    def forward(self, inputs):

        # remember the inputs
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites

    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# class Activation_Softmax:

#     # Forward pass
#     def forward(self, inputs):
#         # Remember input values
#         self.inputs = inputs

#         # Get unnormalized probabilities
#         exp_values = np.exp(inputs - np.max(inputs, axis=1,
#                                             keepdims=True))
#         # Normalize them for each sample
#         probabilities = exp_values / np.sum(exp_values, axis=1,
#                                             keepdims=True)

#         self.output = probabilities

#     # Backward pass
#     def backward(self, dvalues):

#         # Create uninitialized array
#         self.dinputs = np.empty_like(dvalues)

#         # Enumerate outputs and gradients
#         for index, (single_output, single_dvalues) in \
#                 enumerate(zip(self.output, dvalues)):
#             # Flatten output array
#             single_output = single_output.reshape(-1, 1)
#             # Calculate Jacobian matrix of the output
#             jacobian_matrix = np.diagflat(single_output) - \
#                               np.dot(single_output, single_output.T)

#           # Calculate sample-wise gradient
#             # and add it to the array of sample gradients
#             self.dinputs[index] = np.dot(jacobian_matrix,
#                                          single_dvalues)
# =============================================================================
# First, we created an empty array (which will become the resulting gradient array) with the same
# shape as the gradients that we’re receiving to apply the chain rule. The np.empty_like method
# creates an empty and uninitialized array. Uninitialized means that we can expect it to contain
# meaningless values, but we’ll set all of them shortly anyway, so there’s no need for initialization
# (for example, with zeros using np.zeros() instead). In the next step, we’re going to iterate
# sample-wise over pairs of the outputs and gradients, calculating the partial derivatives as
# described earlier and calculating the final product (applying the chain rule) of the Jacobian matrix
# and gradient vector (from the passed-in gradient array), storing the resulting vector as a row in
# the dinput array. We’re going to store each vector in each row while iterating, forming the output
# array.
#
# =============================================================================
# class Loss:

#         # Set/remember trainable layers
#     def remember_trainable_layers(self, trainable_layers):
#         self.trainable_layers = trainable_layers


#     def calculate(self , output , y ):
#         sample_losses = self.forward(output,y)
#         data_loss = np.mean(sample_losses)

#         # self.accumulated_sum += np.sum(sample_losses)
#         # self.accumulated_count += len(sample_losses)

#         return data_loss

# Common loss class
class Loss:

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it

        return data_loss

    # Calculates accumulated loss

    def calculate_accumulated(self):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it

        return data_loss

    # Reset variables for accumulated loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    # def calculate_accumulated(self) :

    #     data_loss = self.accumulated_sum / self.accumulated_count

    #     return data_loss

    # def new_pass(self) :

    #     self.accumulated_sum = 0
    #     self.accumulated_count = 0

# class Loss:

#     # Calculates the data and regularization losses
#     # given model output and ground truth values
#     def calculate(self, output, y):

#         # Calculate sample losses
#         sample_losses = self.forward(output, y)

#         # Calculate mean loss
#         data_loss = np.mean(sample_losses)

#         # Return loss
#         return data_loss


class Loss_CategoricalCrossEntropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(y_pred[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / y_pred
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# class Loss_CategoricalCrossentropy(Loss):

#     # Forward pass
#     def forward(self, y_pred, y_true):

#         # Number of samples in a batch
#         samples = len(y_pred)

#         # Clip data to prevent division by 0
#         # Clip both sides to not drag mean towards any value
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

#         # Probabilities for target values -
#         # only if categorical labels
#         if len(y_true.shape) == 1:
#             correct_confidences = y_pred_clipped[
#                 range(samples),
#                 y_true
#             ]


#         # Mask values - only for one-hot encoded labels
#         elif len(y_true.shape) == 2:
#             correct_confidences = np.sum(
#                 y_pred_clipped * y_true,
#                 axis=1
#             )

#         # Losses
#         negative_log_likelihoods = -np.log(correct_confidences)
#         return negative_log_likelihoods

#     # Backward pass
#     def backward(self, y_pred, y_true):

#         # Number of samples
#         samples = len(y_pred)
#         # Number of labels in every sample
#         # We'll use the first sample to count them
#         labels = len(y_pred[0])

#         # If labels are sparse, turn them into one-hot vector
#         if len(y_true.shape) == 1:
#             y_true = np.eye(labels)[y_true]

#         # Calculate gradient
#         self.dinputs = - y_true / y_pred
#         # Normalize gradient
#         self.dinputs = self.dinputs / samples

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

class Activation_Softmax_Loss_CategoricalCrossentropy:

    # # Creates activation and loss function objects
    # def __init__(self):
    #     self.activation = Activation_Softmax()
    #     self.loss = Loss_CategoricalCrossEntropy()

    # # Forward pass
    # def forward(self, inputs, y_true):
    #     # Output layer's activation function
    #     self.activation.forward(inputs)
    #     # Set the output
    #     self.output = self.activation.output
    #     # Calculate and return loss value
    #     return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = y_pred.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# %%

# check if the previous class is correct

# softmax_outputs = np.array([[0.7, 0.1, 0.2],
#                             [0.1, 0.5, 0.4],
#                             [0.02, 0.9, 0.08]])

# class_target = np.array([0 , 1 , 1])

# def f1():
#     softmax_loss = Activation_Softmax_Loss_CategorialCrossentropy()
#     softmax_loss.backward(softmax_outputs, class_target)
#     dvalues1 = softmax_loss.dinputs

# def f2():
#     activation = Activation_Softmax()
#     activation.output = softmax_outputs
#     loss = Loss_CategoricalCrossEntropy()
#     loss.backward(softmax_outputs, class_target)
#     activation.backward(loss.dinputs)
#     dvalues2 = activation.dinputs

# t1 = timeit(lambda: f1(), number=10000)
# t2 = timeit(lambda: f2(), number=10000)
# print(t2/t1)
# print(t1)
# print(t2)

# %%


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        self.dweights = np.dot(self.inputs.T, dvalues)

        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# class Layer_Dense:

#     # Layer initialization
#     def __init__(self, n_inputs, n_neurons):
#         # Initialize weights and biases
#         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))

#     # Forward pass
#     def forward(self, inputs):
#         # Remember input values
#         self.inputs = inputs
#         # Calculate output values from inputs, weights and biases
#         self.output = np.dot(inputs, self.weights) + self.biases

#     # Backward pass
#     def backward(self, dvalues):
#         # Gradients on parameters
#         self.dweights = np.dot(self.inputs.T, dvalues)
#         self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
#         # Gradient on values
#         self.dinputs = np.dot(dvalues, self.weights.T)


# TODO :undestand if in our backpropagation we put the values from the next layer or the values that we had in this one

# say we have 4 inputs (a sample of 4) and we want 3 neurons. To create the appropriate weights we need
# to create a 4x3 matrix. So we create it by the randn
# print(np.random.normal(loc=0.0, scale=1.0, size=(4,3)))
# Create Dense layer with 2 input features and 3 output values
# dense1 = Layer_Dense(2, 3)

# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLU()

# # Create second Dense layer with 3 input features (as we take output
# # of previous layer here) and 3 output values (output values)
# dense2 = Layer_Dense(3, 3)

# # Create Softmax classifier's combined loss and activation
# loss_activation = Activation_Softmax_Loss_CategorialCrossentropy()

# # Perform a forward pass of our training data through this layer
# dense1.forward(X)

# # Perform a forward pass through activation function
# # takes the output of first dense layer here
# activation1.forward(dense1.output)

# # Perform a forward pass through second Dense layer
# # takes outputs of activation function of first layer as inputs
# dense2.forward(activation1.output)

# # Perform a forward pass through the activation/loss function
# # takes the output of second dense layer here and returns loss
# loss = loss_activation.forward(dense2.output, y)
# # Let's see output of the first few samples:
# print(loss_activation.output[:5])

# # Print loss value
# print('loss:', loss)

# # Calculate accuracy from output of activation2 and targets
# # calculate values along first axis
# predictions = np.argmax(loss_activation.output, axis=1)
# if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)
# accuracy = np.mean(predictions==y)

# # Print accuracy
# print('acc:', accuracy)

# # Backward pass
# loss_activation.backward(loss_activation.output, y)
# dense2.backward(loss_activation.dinputs)
# activation1.backward(dense2.dinputs)
# dense1.backward(activation1.dinputs)

# # Print gradients
# print(dense1.dweights)
# print(dense1.dbiases)
# print(dense2.dweights)
# print(dense2.dbiases)


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

# loss_function = Loss_CategoricalCrossEntropy()
# loss = loss_function.calculate(activation_2.output, y)
# print("Loss:", loss)

# #check the predictions
# predictions = np.argmax(activation_2.output , axis = 1 )

# #if targets are 1 hot encoded we need to change them
# if len(y.shape) == 2 :
#     y = np.argmax(y , axis = 1 )

# accuracy = np.mean(predictions == y)
# print("acc:", accuracy)
# %%
# test accuracy

# softmax_outputs = np.array([[0.7, 0.2, 0.1],
# [0.5, 0.1, 0.4],
# [0.02, 0.9, 0.08]])

# #the correct values
# class_targets = np.array([0, 1, 1])

# #check the predictions
# predictions = np.argmax(softmax_outputs, axis = 1 )

# #if targets are 1 hot encoded we need to change them
# if len(class_targets.shape) == 2 :
#     class_targets = np.argmax(class_targets , axis = 1 )

# accuracy = np.mean(predictions == class_targets)
# print("acc:", accuracy)
# %%
# =============================================================================
#
# up until now all we did was the forward method
# Now we need to optimize our network to perform better
# =============================================================================

# Random search for best weights

# lowest_loss = 9999999 # some initial value
# best_dense1_weights = layer_1.weights.copy()
# best_dense1_biases = layer_1.biases.copy()
# best_dense2_weights = layer_2.weights.copy()
# best_dense2_biases = layer_2.biases.copy()

# print(0.05 * np.random.randn(2, 3))
# for iteration in range(10000):
#     #print(0.05 * np.random.randn(1))
#     #Generate new weights and biases for each iteration
#     layer_1.weights = 0.05 * np.random.randn(2, 3)
#     layer_1.biases = 0.05 * np.random.rand(1, 3)
#     layer_2.weights = 0.05 *  np.random.rand(3, 3)
#     layer_2.biases =  0.05 * np.random.rand(1, 3)

#     #forward pass
#     layer_1.forward(X)
#     activation_1.forward(layer_1.output)
#     layer_2.forward(activation_1.output)
#     activation_2.forward(layer_2.output)

#     #find the loss
#     loss = loss_function.calculate(activation_2.output , y)

#     #Calculate accuracy
#     predictions = np.argmax(activation_2.output , axis= 1)
#     accuracy = np.mean(predictions == y)

#     #if loss smaller than original
#     if loss < lowest_loss:
#         print("New set of weights found in iteration:", iteration, "loss:" , loss , "acc:", accuracy)
#         best_dense1_weights = layer_1.weights.copy()
#         best_dense1_biases = layer_1.biases.copy()
#         best_dense2_weights = layer_2.weights.copy()
#         best_dense2_biases = layer_2.biases.copy()
#         lowest_loss = loss

# =============================================================================
# Random is not working
# But we can change it a bit so it can work.
# We start from low values and we work our way up
# if we find a good loss function then we keep it
# if the next one is better then we change it
# otherwise we keep the old one and work close to this value
# =============================================================================
# for iteration in range(10000):
#     #print(0.05 * np.random.randn(1))
#     #Generate new weights and biases for each iteration
#     #we start from really small values and work up
#     layer_1.weights += 0.05 * np.random.randn(2, 3)
#     layer_1.biases += 0.05 * np.random.rand(1, 3)
#     layer_2.weights += 0.05 *  np.random.rand(3, 3)
#     layer_2.biases +=  0.05 * np.random.rand(1, 3)

#     #forward pass
#     layer_1.forward(X)
#     activation_1.forward(layer_1.output)
#     layer_2.forward(activation_1.output)
#     activation_2.forward(layer_2.output)

#     #find the loss
#     loss = loss_function.calculate(activation_2.output , y)

#     #Calculate accuracy
#     predictions = np.argmax(activation_2.output , axis= 1)
#     accuracy = np.mean(predictions == y)

#     #if loss smaller than original
#     if loss < lowest_loss:
#         print("New set of weights found in iteration:", iteration, "loss:" , loss , "acc:", accuracy)
#         best_dense1_weights = layer_1.weights.copy()
#         best_dense1_biases = layer_1.biases.copy()
#         best_dense2_weights = layer_2.weights.copy()
#         best_dense2_biases = layer_2.biases.copy()
#         lowest_loss = loss
#     else:
#         #we start from really small values and work up
#         layer_1.weights = best_dense1_weights.copy()
#         layer_1.biases = best_dense1_biases.copy()
#         layer_2.weights = best_dense2_weights.copy()
#         layer_2.biases =  best_dense2_biases.copy()

# =============================================================================
# This method is working for really simple problems but it doesnt work for more complicated ones
# we need to find a better way to optimize our neural network
# =============================================================================

# %%
# # Passed-in gradient from the next layer
# # for the purpose of this example we're going to use
# # an array of an incremental gradient values
# dvalues = np.array([[1., 1., 1.],
# [2., 2., 2.],
# [3., 3., 3.]])
# # We have 3 sets of weights - one set for each neuron
# # we have 4 inputs, thus 4 weights
# # recall that we keep weights transposed
# weights = np.array([[0.2, 0.8, -0.5, 1],
# [0.5, -0.91, 0.26, -0.5],
# [-0.26, -0.27, 0.17, 0.87]]).T

# # and multiply by the passed-in gradient for this neuron
# #dinputs is the partial derivative of the output with respect to inputs

# dinputs = np.dot(dvalues, weights.T)
# print(dinputs)

# # We have 3 sets of inputs - samples
# inputs = np.array([[1, 2, 3, 2.5],
# [2., 5., -1., 2],
# [-1.5, 2.7, 3.3, -0.8]])
# # sum weights of given input
# # and multiply by the passed-in gradient for this neuron
# #dweight is the partial derivative of the output with respect to weights
# # dweights = np.dot(inputs.T, dvalues)
# # print(dweights)

# # # One bias for each neuron
# # # biases are the row vector with a shape (1, neurons)
# biases = np.array([[2, 3, 0.5]])
# # # dbiases - sum values, do this over samples (first axis), keepdims
# # # since this by default will produce a plain list -
# # # we explained this in the chapter 4
# # #we take the sum of the clumns as we do with the dot products
# # dbiases = np.sum(dvalues, axis=0, keepdims=True)
# # print(dbiases)

# # Forward pass
# layer_outputs = np.dot(inputs, weights) + biases # Dense layer
# relu_outputs = np.maximum(0, layer_outputs) # ReLU activation
# # Let's optimize and test backpropagation here
# # ReLU activation - simulates derivative with respect to input values
# # from next layer passed to current layer during backpropagation
# drelu = relu_outputs.copy()
# drelu[layer_outputs <= 0] = 0
# # EXPLAINATION
# # layer_outputs <= 0 Checks which values of layer_ouputs are <= 0 and labels them with true or false

# # Dense layer
# # dinputs - multiply by weights
# dinputs = np.dot(drelu, weights.T)
# # dweights - multiply by inputs
# dweights = np.dot(inputs.T, drelu)
# # dbiases - sum values, do this over samples (first axis), keepdims
# # since this by default will produce a plain list -
# # we explained this in the chapter 4
# dbiases = np.sum(drelu, axis=0, keepdims=True)
# # Update parameters
# weights += -0.001 * dweights
# biases += -0.001 * dbiases
# print(weights)
# print(biases)


# %%
# #Softmax derivative

# softmax_output = [ 0.7 , 0.1 , 0.2 ]
# softmax_output = np.array(softmax_output)
# print(softmax_output.shape)

# # =============================================================================
# # In the context of reshaping arrays in NumPy, the -1 in the reshape method serves as a placeholder
# # that allows NumPy to automatically compute the size of that particular dimension.
# # The goal is to make the reshaping operation convenient, especially when the size of one dimension
# # is not explicitly specified.
# # =============================================================================
# #this must happen for the dot product to be correct size
# softmax_output = np.array(softmax_output).reshape(-1, 1)
# print(softmax_output.shape)
# print(softmax_output)
# # print(softmax_output)
# print(np.diagflat(softmax_output))
# # print(np.diagflat(softmax_output))

# print(np.dot(softmax_output, softmax_output.T))

# print(np.diagflat(softmax_output)- np.dot(softmax_output , softmax_output.T))

# %%
# =============================================================================
#
# The work of the optimizer it to change the weights after the backpropagation.
# we know the amount that we need to change the weight so now we have to make this calcualtion
# to do so we will use the stochastic gradient descent. What it does is it substracts from the layer weights
# the gradient of the weigth (the partial derivative that we found) and from the biases ny multiplying it with
# a learning rate which for now will be 1.
# it is one of th esimplest optimizers to use
# =============================================================================
# =============================================================================
#
# we can start with big learning rate and the change the leraning raet as we move forward.
# This way we can start with bigger step towarsd global minimum (and get away from local minimums)
# and as we go towards our goal, decrease the learning rate so that it find the global minimum
#
# =============================================================================
class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1. + self.decay * self.iterations)

    def update_params(self, layer):

        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):

                layer.weight_momentums = np.zeros_like(layer.weights)

                layer.bias_momentums = np.zeros_like(layer.biases)

            # takes a percentage of the previous weight and adds it to the current change of weight
            # this way the momentum from the prev changes impact the new changes so that we dont get stuck in
            # a local minimum
            layer.weight_momentums = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights

            layer.weights += layer.weight_momentums

            layer.bias_momentums = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.biases += layer.bias_momentums

        else:
            layer.weights += -self.current_learning_rate * layer.dweights
            layer.biases += -self.current_learning_rate * layer.dbiases

    def post_update_params(self):

        self.iterations += 1


# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) +
             self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) +
             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# # SGD optimizer
# class Optimizer_SGD:

#     # Initialize optimizer - set settings,
#     # learning rate of 1. is default for this optimizer
#     def __init__(self, learning_rate=1.0):
#         self.learning_rate = learning_rate

#     # Update parameters
#     def update_params(self, layer):
#         layer.weights += -self.learning_rate * layer.dweights
#         layer.biases += -self.learning_rate * layer.dbiases

# optimizer = Optimizer_SGD()

# optimizer.update_params(dense1)
# optimizer.update_params(dense2)

# dense1 = Layer_Dense(2, 64)
# dense2 = Layer_Dense(64 , 3)


# Create Dense layer with 2 input features and 64 output values
# dense1 = Layer_Dense(X.shape[1], 128)

# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLU()
# activation2 = Activation_ReLU()
# # Create second Dense layer with 64 input features (as we take output
# # of previous layer here) and 3 output values (output values)
# dense2 = Layer_Dense(128, 128)

# dense3 = Layer_Dense(128, 10)

# # Create Softmax classifier's combined loss and activation
# loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


# # Create optimizer
# optimizer = Optimizer_SGD(learning_rate=1 , decay = 1e-3 , momentum=0.8) #2.5 has 83% accuracy
# # learning_rate=2 , decay = 1e-3 , momentum=0.5 --> 88.7 % accuracy
# # learning_rate=1.5 , decay = 1e-3 , momentum=0.8 --> 91 % accurcay 128 hidden neurons

# # Train in loop
# for epoch in range(1 , EPOCHS+1):

#     print(f'epoch:{epoch}')
#     # Perform a forward pass of our training data through this layer

#     for step in range(steps):
#         batch_X = X[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
#         batch_y = y[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
#         dense1.forward(batch_X)

#         # Perform a forward pass through activation function
#         # takes the output of first dense layer here
#         activation1.forward(dense1.output)

#         # Perform a forward pass through second Dense layer
#         # takes outputs of activation function of first layer as inputs
#         dense2.forward(activation1.output)
#         activation2.forward(dense2.output)

#         dense3.forward(activation2.output)
#         # Perform a forward pass through the activation/loss function
#         # takes the output of second dense layer here and returns loss
#         loss = loss_activation.forward(dense3.output, batch_y)
#         # Calculate accuracy from output of activation2 and targets
#         # calculate values along first axis
#         predictions = np.argmax(loss_activation.output, axis=1)
#         if len(batch_y.shape) == 2:
#             batch_y = np.argmax(batch_y, axis=1)


#         accuracy = np.mean(predictions==batch_y)


#         print(f'epoch: {epoch}, ' +
#               f'acc: {accuracy:.3f}, ' +
#               f'loss: {loss:.3f}')

#         # Backward pass
#         loss_activation.backward(loss_activation.output, batch_y)

#         dense3.backward(loss_activation.dinputs)
#         activation2.backward(dense3.dinputs)

#         dense2.backward(activation2.dinputs)

#         activation1.backward(dense2.dinputs)
#         dense1.backward(activation1.dinputs)

#         # Update weights and biases
#         optimizer.update_params(dense1)
#         optimizer.update_params(dense2)
#         optimizer.update_params(dense3)


# # Validate the model

# # Create test dataset
# # X_test, y_test = spiral_data(samples=100, classes=3)

# # Perform a forward pass of our testing data through this layer
# dense1.forward(X_test)

# # Perform a forward pass through activation function
# # takes the output of first dense layer here
# activation1.forward(dense1.output)

# # Perform a forward pass through second Dense layer
# # takes outputs of activation function of first layer as inputs
# dense2.forward(activation1.output)

# activation2.forward(dense2.output)
# dense3.forward(activation2.output)
# # Perform a forward pass through the activation/loss function
# # takes the output of second dense layer here and returns loss
# loss = loss_activation.forward(dense3.output, y_test)

# # Calculate accuracy from output of activation2 and targets
# # calculate values along first axis
# predictions = np.argmax(loss_activation.output, axis=1)
# if len(y_test.shape) == 2:
#     y_test = np.argmax(y_test, axis=1)
# accuracy = np.mean(predictions==y_test)

# print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


# %%

class Accuracy:

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return accuracy
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Layer_Input:

    def forward(self, inputs):
        self.output = inputs


class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

# X, y = sine_data()
# Model class
# class Model:

#     def __init__(self):
#         # Create a list of network objects
#         self.layers = []

#     # Add objects to the model
#     def add(self, layer):
#         self.layers.append(layer)

#     # Set loss and optimizer
#     def set(self, *, loss, optimizer):
#         self.loss = loss
#         self.optimizer = optimizer

#     # Finalize the model
#     def finalize(self):

#         # Create and set the input layer
#         self.input_layer = Layer_Input()

#         # Count all the objects
#         layer_count = len(self.layers)

#         # Iterate the objects
#         for i in range(layer_count):

#             # If it's the first layer,
#             # the previous layer object is the input layer
#             if i == 0:
#                 self.layers[i].prev = self.input_layer
#                 self.layers[i].next = self.layers[i+1]

#             # All layers except for the first and the last
#             elif i < layer_count - 1:
#                 self.layers[i].prev = self.layers[i-1]
#                 self.layers[i].next = self.layers[i+1]

#             # The last layer - the next object is the loss
#             else:
#                 self.layers[i].prev = self.layers[i-1]
#                 self.layers[i].next = self.loss

#     # Train the model
#     def train(self, X, y, *, epochs=1, print_every=1):

#         # Main training loop
#         for epoch in range(1, epochs+1):

#             # Perform the forward pass
#             output = self.forward(X)

#             # Temporary
#             print(output)
#             exit()

#     # Performs forward pass
#     def forward(self, X):

#         # Call forward method on the input layer
#         # this will set the output property that
#         # the first layer in "prev" object is expecting
#         self.input_layer.forward(X)

#         # Call forward method of every object in a chain
#         # Pass output of the previous object as a parameter
#         for layer in self.layers:
#             layer.forward(layer.prev.output)

#         # "layer" is now the last object from the list,
#         # return its output
#         return layer.output


# we will create a class model to sum up all the classes so far and make it simpler to use
# Model class
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss, optimizer and accuracy

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # the input layer which is our data
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        # maybe this is not needed
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i == layer_count - 1:

                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(
            self.trainable_layers
        )

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        # this is if we have the time to create
        # more types of activations and losses
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossEntropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # If there is validation data passed,
        # set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1

            # For better readability
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                # Dividing rounds down. If there are some remaining
                # data but nor full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss,  = self.loss.calculate(output, batch_y,)
                loss = data_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                    output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          #   f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss = self.loss.calculate_accumulated()
            epoch_loss = epoch_data_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  #   f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            # If there is the validation data
            if validation_data is not None:

                # Reset accumulated values in loss
                # and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()

                # Iterate over steps
                for step in range(validation_steps):

                    # If batch size is not set -
                    # train using one step and full dataset
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val

                    # Otherwise slice a batch
                    else:
                        batch_X = X_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                        batch_y = y_val[
                            step*batch_size:(step+1)*batch_size
                        ]

                    # Perform the forward pass
                    output = self.forward(batch_X, training=False)

                    # Calculate the loss
                    self.loss.calculate(output, batch_y)

                    # Get predictions and calculate an accuracy
                    predictions = self.output_layer_activation.predictions(
                        output)
                    self.accuracy.calculate(predictions, batch_y)

                # Get and print validation loss and accuracy
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                # Print a summary
                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    # Performs backward pass

    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 2))
model.add(Accuracy_Categorical())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
# Train the model
model.train(X_train, y_train, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)
