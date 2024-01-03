# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 19:52:31 2024

@author: iason
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

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

#%% netpune 
import neptune

run = neptune.init_run(
    project="jason-k/svm-neural",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZTI4NWM4OC0wMDg2LTQ2YTItYmFmMi1iZGQ3MmZhN2U5MDkifQ==",
)  # your credentials

#%% create neptune model 

model_version = neptune.init_model_version(
    model="SVMNEUR-SELFMADE",
    project="jason-k/svm-neural",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZTI4NWM4OC0wMDg2LTQ2YTItYmFmMi1iZGQ3MmZhN2U5MDkifQ==", # your credentials
)


#%% 

from sklearn.cluster import KMeans

def gaussian_rbf(x, center, sigma):
    return np.exp(-cdist(x, center, 'sqeuclidean') / (2 * sigma**2))


def random_centers(n_centers , X_train ):
    center_indices = np.random.choice(X_train.shape[0], n_centers, replace=False)
    rbf_centers = X_train[center_indices]
    return rbf_centers 

def K_means_centers(X , max_iters , n_centers): 
    
    km = KMeans(n_clusters=n_centers, max_iter=max_iters, verbose=0 )
    print('here')
    km.fit(X)
    print('hey')
    return km.cluster_centers_

rbf_width = 10.0

def rbf_layer(X, rbf_centers, rbf_width):
    return gaussian_rbf(X, rbf_centers, rbf_width)

def rbfn_predict(X, rbf_centers, rbf_width, weights):
    rbf_outputs = rbf_layer(X, rbf_centers, rbf_width)
    return np.dot(rbf_outputs,weights)


#%% 
rbf_centers = random_centers(n_centers=500 , X_train = X_train )

centers = [ 10 , 50 , 100 , 200 , 500]

#k_centers = K_means_centers(X = X_batch , max_iters = 300 , n_centers = 100)
print('hi')
rbf_outputs_train = rbf_layer(X_train, rbf_centers, rbf_width)

# Perform linear regression
from sklearn.linear_model import LogisticRegression

# Train logistic regression model
logistic_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs' , max_iter=1000)

# Training loop
# Train the model
logistic_regression.fit(rbf_outputs_train, y_train)
#%% 
print("Number of iterations:", logistic_regression.n_iter_)
print(logistic_regression.n_features_in_)
#print(logistic_regression.feature_names_in_)
print(logistic_regression.classes_)
print(logistic_regression.coef_)
# =============================================================================
#  In logistic regression, the coefficients (coef_) represent the weights assigned 
#  to each feature in the model. These weights are used in the logistic function 
#  to calculate the log-odds of the target variable being in a particular class.
# =============================================================================
print('intercept' , logistic_regression.intercept_) #biases


# Log accuracy at specified intervals
#%% test data
rbf_outputs_test = rbf_layer(X_test, rbf_centers, rbf_width)
y_pred = logistic_regression.predict(rbf_outputs_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

