# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:15:59 2024

@author: iason
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns # for statistical data visualization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#%% Gaussin class 

class GaussianRBF:
    def __init__(self):
        pass

    def gaussian_rbf(self, x, center, sigma):
        return np.exp(-cdist(x, center, 'sqeuclidean') / (2 * sigma**2))


class CentersInitializer:
    def __init__(self):
        pass

    def random_centers(self, n_centers, X_train):
        center_indices = np.random.choice(X_train.shape[0], n_centers, replace=False)
        rbf_centers = X_train[center_indices]
        return rbf_centers

    def k_means_centers(self, X, max_iters, n_centers):
        km = KMeans(n_clusters=n_centers, max_iter=max_iters, verbose=0)
        km.fit(X)
        return km.cluster_centers_


class RBFNeuralNetwork(GaussianRBF, CentersInitializer):
    def __init__(self, n_centers, rbf_width):
        self.n_centers = n_centers
        self.rbf_width = rbf_width
        self.rbf_centers = None
        self.logistic_regression = None

    def fit(self, X_train, y_train, center_initializer='random'):
        # Choose or initialize RBF centers
        if center_initializer == 'random':
            self.rbf_centers = self.random_centers(n_centers=self.n_centers, X_train=X_train)
        elif center_initializer == 'k_means':
            self.rbf_centers = self.k_means_centers(X_train, max_iters=300, n_centers=self.n_centers)

        # Calculate RBF layer outputs
        rbf_outputs_train = self.gaussian_rbf(X_train, self.rbf_centers, self.rbf_width)

        # Train logistic regression model
        self.logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        self.logistic_regression.fit(rbf_outputs_train, y_train)

    def predict(self, X_test):
        # Calculate RBF layer outputs for test data
        rbf_outputs_test = self.gaussian_rbf(X_test, self.rbf_centers, self.rbf_width)

        # Make predictions using logistic regression
        return self.logistic_regression.predict(rbf_outputs_test)

#%% 

class GaussianRBF_torch:
    def __init__(self):
        pass

    @staticmethod
    def gaussian_rbf(x, center, sigma):
        return torch.exp(-torch.cdist(x, center, p=2) / (2 * sigma**2))


class RBFNeuralNetwork_torch(GaussianRBF_torch, CentersInitializer):
    def __init__(self, n_centers, rbf_width):
        self.n_centers = n_centers
        self.rbf_width = rbf_width
        self.rbf_centers = None
        self.logistic_regression = None

    def fit(self, X_train, y_train, center_initializer='random', lr=0.01, epochs=100 , weight_decay = 1e-3 , optimizer = 'adam'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if center_initializer == 'random':
            self.rbf_centers = torch.tensor(self.random_centers(n_centers=self.n_centers, X_train=X_train),
                                            dtype=torch.float32, device=device)
        elif center_initializer == 'k_means':
            self.rbf_centers = torch.tensor(self.k_means_centers(X_train, max_iters=300, n_centers=self.n_centers),
                                            dtype=torch.float32, device=device)

        rbf_outputs_train = self.gaussian_rbf(torch.tensor(X_train, dtype=torch.float32, device=device),
                                              self.rbf_centers, self.rbf_width)

        input_dim = rbf_outputs_train.shape[1]
        output_dim = len(np.unique(y_train))

        model = nn.Linear(input_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        if optimizer == 'adam': 
            optimizer = optim.Adam(model.parameters(), lr=lr , weight_decay= weight_decay)
        elif optimizer == 'sgd' : 
            optimizer = optim.SGD(model.parameters(), lr=lr)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                rbf_outputs = self.gaussian_rbf(inputs, self.rbf_centers, self.rbf_width)
                logits = model(rbf_outputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    rbf_outputs_test = self.gaussian_rbf(torch.tensor(X_train, dtype=torch.float32, device=device),
                                                         self.rbf_centers, self.rbf_width)
                    logits_test = model(rbf_outputs_test)
                    y_pred = torch.argmax(logits_test, dim=1).cpu().numpy()
                    accuracy = accuracy_score(y_train, y_pred)
                    print(f"Epoch {epoch}, Accuracy: {accuracy * 100:.2f}%")

        self.logistic_regression = model

    def predict(self, X_test):
        rbf_outputs_test = self.gaussian_rbf(torch.tensor(X_test, dtype=torch.float32),
                                             self.rbf_centers, self.rbf_width)
        logits = self.logistic_regression(rbf_outputs_test)
        return torch.argmax(logits, dim=1).cpu().numpy()


#%% 
from tensorflow.keras.datasets import cifar10  # to import our data
import random


# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# make the rgb value from 0 - 255 --> 0 - 1 ==> scaling
X_train, X_test = X_train / 255.0, X_test / 255.0

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# from 1 col mul rows --> 1 row mul cols
print(y_train)
y_train = y_train.reshape(-1,)
print(y_train)

print(y_test)
y_test = y_test.reshape(-1,)
print(y_test)

# test to see if it works properly


def showImage(x, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(x[index])
    plt.xlabel(class_names[y[index]])


showImage(X_train, y_train, random.randint(0, 9))

# the train and test data
print(X_train.shape, X_test.shape)

# Shuffle the training dataset
keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]

#%% reshape the data 

# we want to reshape the image from a 4D array to a 2D array
# This line extracts the number of samples, image height, image width, and number of channels (e.g., RGB channels) from the shape of the X_train array.
num_samples, img_height, img_width, num_channels = X_train.shape

# it flattens the image data, converting each image into a one-dimensional vector.
# The resulting shape is (num_samples, img_height * img_width * num_channels),
X_train = X_train.reshape(num_samples, -1)
num_samples, img_height, img_width, num_channels = X_test.shape
X_test = X_test.reshape(num_samples, -1)
 

#%% batching instead of 2 cateogires 

batch_size = 10000
X_batch = X_train[:batch_size]
y_batch = y_train[:batch_size]

batch_size_test = 2000
X_batch_test = X_test[:batch_size_test]
y_batch_test = y_test[:batch_size_test]

#%% custom made rbf nn 

centers = [10 , 50 , 100 , 200 , 500]
variance = [ 0.5 , 1 , 5 , 8 , 10 , 20 ]

rbf_nn = RBFNeuralNetwork(n_centers=500, rbf_width=10.0)
rbf_nn.fit(X_train, y_train, center_initializer='random')

#%% pytorch

rbf_nn_tensor = RBFNeuralNetwork_torch(n_centers=100, rbf_width=8.0)
rbf_nn_tensor.fit(X_train, y_train, center_initializer='random', lr=0.05, epochs=100 , optimizer='adam')

#%% predictions

# Make predictions on test set
y_pred = rbf_nn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy * 100:.2f}%")

print(classification_report(y_test, y_pred))


#%% confusion matrix 

cm = confusion_matrix(y_test, y_pred)
class_names_selected = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names_selected, yticklabels=class_names_selected)
heatmap_1.set_title("confusion matrix for neural network TEST")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()