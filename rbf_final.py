# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:15:59 2024

@author: iasonas kakandris 
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns # for statistical data visualization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.decomposition import PCA

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
    
    def k_means_centers_per_class(self, X_train, y_train, max_iters, n_centers):
        unique_classes = np.unique(y_train)
        class_centers = {}
    
        for class_label in unique_classes:
            # Extract data for the current class
            class_data = X_train[y_train == class_label]
    
            # Apply k-means
            km = KMeans(n_clusters=n_centers, max_iter=max_iters, verbose=0)
            km.fit(class_data)
    
            # Save cluster centers
            class_centers[class_label] = km.cluster_centers_
    
        return class_centers


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
        elif center_initializer == 'k_means_per_class':
          class_centers = self.k_means_centers_per_class(X_train, y_train, max_iters=300, n_centers=self.n_centers)
          # Use the class centers as the RBF centers
          self.rbf_centers = torch.tensor(np.concatenate(list(class_centers.values())), dtype=torch.float32, device=device)

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
        dataloader = DataLoader(dataset, batch_size=128 , shuffle=True) #, batch_size=128

        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                rbf_outputs = self.gaussian_rbf(inputs, self.rbf_centers, self.rbf_width)
                logits = model(rbf_outputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0 or epoch == (epochs - 1) :
                with torch.no_grad():
                    rbf_outputs_test = self.gaussian_rbf(torch.tensor(X_train, dtype=torch.float32, device=device),
                                                         self.rbf_centers, self.rbf_width)
                    logits_test = model(rbf_outputs_test)
                    y_pred = torch.argmax(logits_test, dim=1).cpu().numpy()
                    accuracy = accuracy_score(y_train, y_pred)
                    print(f"Epoch {epoch}, Accuracy: {accuracy * 100:.2f}%")
                    # Calculate and print the loss
                    loss_value = criterion(logits_test, torch.tensor(y_train, dtype=torch.long, device=device)).item()
                    print(f"Epoch {epoch}, Loss: {loss_value:.4f}")
                    cm = confusion_matrix(y_train, y_pred)
                    class_names = ['airplane', 'automobile', 'bird', 'cat',
                                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

                        
                    heatmap_1 = sns.heatmap(
                         cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
                    heatmap_1.set_title("confusion matrix")
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                     # print(classification_report(batch_y, output_for_matrix))
                    plt.show()


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

#%% pca 

# Initialize PCA
pca = PCA()

# Fit on training data
pca.fit(X_train)

# Plot the cumulative explained variance ratio
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()
 
#%% apply pca

# Apply PCA
num_components = 500  # You can choose the number of components based on your experimentation
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#%% restore to original shape ( if needed ) 

# Reshape back to the original format
X_train = X_train_pca.reshape(num_samples, img_height, img_width, num_channels)
X_test = X_test_pca.reshape(num_samples, img_height, img_width, num_channels)
#%% variance ratio 


print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance:", np.sum(pca.explained_variance_ratio_))

#%% batching instead of 2 cateogires 

batch_size = 10000
X_batch = X_train[:batch_size]
y_batch = y_train[:batch_size]

batch_size_test = 2000
X_batch_test = X_test[:batch_size_test]
y_batch_test = y_test[:batch_size_test]

#%% 
print(X_train_pca.shape)
print(X_train.shape)
print(X_test_pca.shape)
#%% custom made rbf nn 

centers = [10 , 50 , 100 , 200 , 500]
variance = [ 0.5 , 1 , 5 , 8 , 10 , 20 ]

rbf_nn = RBFNeuralNetwork(n_centers=500, rbf_width=10)

start_time = time.perf_counter()
rbf_nn.fit(X_train_pca, y_train, center_initializer='k_means')
end_time = time.perf_counter()

# confusion matrix for train data 
y_pred_train = rbf_nn.predict(X_train_pca)
cm = confusion_matrix(y_train, y_pred_train)
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    
heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
heatmap_1.set_title("confusion matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()

# accuracy for train data 
accuracy = accuracy_score(y_train, y_pred_train) 
print(f"Accuracy: {accuracy * 100:.2f}%")


y_pred = rbf_nn.predict(X_test_pca)

cm = confusion_matrix(y_test, y_pred)
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    
heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
heatmap_1.set_title("confusion matrix test")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()

accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy * 100:.2f}%")



print('elapsed time ' , end_time-start_time)
print(classification_report(y_test, y_pred))


#%% pytorch

rbf_nn_tensor = RBFNeuralNetwork_torch(n_centers=500, rbf_width=2.0) # sigma = 2.0 sigma 
start_time = time.perf_counter()
rbf_nn_tensor.fit(X_train_pca, y_train, center_initializer='k_means_per_class', lr=0.008, epochs=150 , optimizer='adam' , weight_decay = 1e-6)
end_time = time.perf_counter() #lr 0.01 weight = 1e-3 really good , lr = 0.01 weight = 1e-2 not good, lr= 0.05 weight 1e-3 really good 38.15%
                                # lt 0.005 weight 1e-4 39.23% k_means_per_class  lt 0.005 weight 1e-5 43% k_means_per_class lt 0.008 weight 1e-6 45.29% k_means_per_class

# predictions


# Make predictions on test set
y_pred = rbf_nn_tensor.predict(X_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy * 100:.2f}%")

print(classification_report(y_test, y_pred))
print('elapsed time ' , end_time-start_time)
# confusion matrix 

cm = confusion_matrix(y_test, y_pred)
class_names_selected = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
heatmap_1.set_title("confusion matrix for neural network TEST")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()