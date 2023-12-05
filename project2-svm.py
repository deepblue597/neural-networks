# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:04:37 2023

@author: IASON
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
from sklearn.preprocessing import StandardScaler

# import SVC classifier
from sklearn.svm import SVC

# import metrics to compute accuracy
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from tensorflow.keras.datasets import cifar10  # to import our data

import random



#%% netpune 
import neptune
#from neptune.version import version as neptune_client_version
#project = neptune.init_project(project="jason-k/example-project-tensorflow-keras", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZTI4NWM4OC0wMDg2LTQ2YTItYmFmMi1iZGQ3MmZhN2U5MDkifQ==")

run = neptune.init_run(
    project="jason-k/svm-neural",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZTI4NWM4OC0wMDg2LTQ2YTItYmFmMi1iZGQ3MmZhN2U5MDkifQ==",
)  # your credentials

#%% CIFAR 10 dataset 

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



# # we want to reshape the image from a 4D array to a 2D array
# # This line extracts the number of samples, image height, image width, and number of channels (e.g., RGB channels) from the shape of the X_train array.
# num_samples, img_height, img_width, num_channels = X_train.shape

# # it flattens the image data, converting each image into a one-dimensional vector.
# # The resulting shape is (num_samples, img_height * img_width * num_channels),
# X_train = X_train.reshape(num_samples, -1)
# num_samples, img_height, img_width, num_channels = X_test.shape
# X_test = X_test.reshape(num_samples, -1)

#%%  Select two classes (e.g., 'airplane' and 'automobile')

class1, class2 = 0, 1  # You can choose the class indices based on the CIFAR-10 class names

# Filter training data and labels for the selected classes
selected_train_indices = np.where((y_train == class1) | (y_train == class2))[0]
X_train_selected = X_train[selected_train_indices]
y_train_selected = y_train[selected_train_indices]

# Filter test data and labels for the selected classes
selected_test_indices = np.where((y_test == class1) | (y_test == class2))[0]
X_test_selected = X_test[selected_test_indices]
y_test_selected = y_test[selected_test_indices]


# Print the shape of the filtered datasets
print("Shape of filtered training data:", X_train_selected.shape)
print("Shape of filtered training labels:", y_train_selected.shape)
print("Shape of filtered test data:", X_test_selected.shape)
print("Shape of filtered test labels:", y_test_selected.shape)

# we want to reshape the image from a 4D array to a 2D array
# This line extracts the number of samples, image height, image width, and number of channels (e.g., RGB channels) from the shape of the X_train array.
num_samples, img_height, img_width, num_channels = X_train_selected.shape

# it flattens the image data, converting each image into a one-dimensional vector.
# The resulting shape is (num_samples, img_height * img_width * num_channels),
X_train_selected = X_train_selected.reshape(num_samples, -1)
num_samples, img_height, img_width, num_channels = X_test_selected.shape
X_test_selected = X_test_selected.reshape(num_samples, -1)

#%% Sigmoid kernel 

# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 


# fit classifier to training set
sigmoid_svc.fit(X_train_selected,y_train_selected)

# make predictions on test set
y_pred=sigmoid_svc.predict(X_test_selected)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test_selected, y_pred)))

#Model accuracy score with sigmoid kernel and C=1.0 : 0.6925

# visualize confusion matrix with seaborn heatmap
cm = confusion_matrix(y_test_selected, y_pred)
class_names_selected = ['airplane', 'automobile']

    
heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names_selected, yticklabels=class_names_selected)
heatmap_1.set_title("confusion matrix for neural network TEST")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()


#%% Polynomial kernel 

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 
# degree int, default=3 

# fit classifier to training set
poly_svc.fit(X_train_selected,y_train_selected)


# make predictions on test set
y_pred=poly_svc.predict(X_test_selected)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test_selected, y_pred)))

#Model accuracy score with polynomial kernel and C=1.0 : 0.9145

#%% Confusion matrix 

# visualize confusion matrix with seaborn heatmap
cm = confusion_matrix(y_test_selected, y_pred)
class_names_selected = ['airplane', 'automobile']

    
heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names_selected, yticklabels=class_names_selected)
heatmap_1.set_title("confusion matrix for neural network TEST")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()

#sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#%%  Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.

# instantiate classifier with default hyperparameters
svc=SVC() 


# fit classifier to training set
svc.fit(X_train_selected,y_train_selected)

# make predictions on test set
y_pred=svc.predict(X_test_selected)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test_selected, y_pred)))
#Model accuracy score with default hyperparameters: 0.9040

cm = confusion_matrix(y_test_selected, y_pred)
class_names_selected = ['airplane', 'automobile']

heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names_selected, yticklabels=class_names_selected)
heatmap_1.set_title("confusion matrix for neural network TEST")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()

#%% linear kernel 


# instantiate classifier with polynomial kernel and C=1.0
linear_svc=SVC(kernel='linear') 
# degree int, default=3 

# fit classifier to training set
linear_svc.fit(X_train_selected,y_train_selected)


# make predictions on test set
y_pred=linear_svc.predict(X_test_selected)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test_selected, y_pred)))

#Model accuracy score with polynomial kernel and C=1.0 : 0.9145

cm = confusion_matrix(y_test_selected, y_pred)
class_names_selected = ['airplane', 'automobile']

heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names_selected, yticklabels=class_names_selected)
heatmap_1.set_title("confusion matrix for neural network TEST")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()



#%% batching instead of 2 cateogires 

batch_size = 10000
X_batch = X_train[:batch_size]
y_batch = y_train[:batch_size]

batch_size_test = 2000
X_batch_test = X_test[:batch_size_test]
y_batch_test = y_test[:batch_size_test]


# we want to reshape the image from a 4D array to a 2D array
# This line extracts the number of samples, image height, image width, and number of channels (e.g., RGB channels) from the shape of the X_train array.
num_samples, img_height, img_width, num_channels = X_batch.shape

# it flattens the image data, converting each image into a one-dimensional vector.
# The resulting shape is (num_samples, img_height * img_width * num_channels),
X_batch = X_batch.reshape(num_samples, -1)
num_samples, img_height, img_width, num_channels = X_batch_test.shape
X_batch_test = X_batch_test.reshape(num_samples, -1)

#%% Polynomial kernel for batch 

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 
# degree int, default=3 

# fit classifier to training set
poly_svc.fit(X_batch,y_batch)


# make predictions on test set
y_pred=poly_svc.predict(X_batch_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_batch_test, y_pred)))

#Model accuracy score with polynomial kernel and C=1.0 : 0.9145

cm = confusion_matrix(y_test_selected, y_pred)
class_names_selected = ['airplane', 'automobile']

heatmap_1 = sns.heatmap(
     cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names_selected, yticklabels=class_names_selected)
heatmap_1.set_title("confusion matrix for neural network TEST")
plt.xlabel('Predicted')
plt.ylabel('Actual')
 # print(classification_report(batch_y, output_for_matrix))
plt.show()

#%%  Hyperparameter Optimization using GridSearch CV

# import GridSearchCV
from sklearn.model_selection import GridSearchCV


# import SVC classifier
from sklearn.svm import SVC


# instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
svc=SVC() 



# declare parameters for hyperparameter tuning
parameters = [ {'C':[1, 10], 'kernel':['linear']},
               {'C':[1, 10], 'kernel':['rbf'], 'gamma':[0.1, 0.5, 0.9]},
               {'C':[1, 10], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.05]} 
              ]


#, 100, 1000
#, 100, 1000  0.2, 0.3, 0.4,  0.6, 0.7, 0.8,
#, 100, 1000  0.02,0.03,0.04,

grid_search = GridSearchCV(estimator = svc,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)

#%% fit the data
grid_search.fit(X_batch, y_batch)

#%% best model 

# examine the best model


# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))


# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

#%% test data in gridsearch
# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_batch_test, y_batch_test)))

#%% self made svm 

class SVM_from_scratch : 
    
    def __init__(self , learning_rate = 0.001 , lambda_param = 0.01 , n_iters = 1000): 
        self.learning_rate = learning_rate 
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None 
        
    def fit(self , X , y): 
        n_samples , n_features = X.shape  
        
        y_ = np.where(y <= 0 , -1 , 1)
        
        #init weights 
        self.weights = np.zeros(n_features) 
        self.bias = 0  
        
        for _ in range(self.n_iters): 
            for index , x_i in enumerate(X): 
                condition = y_[index] * (np.dot(x_i , self.weights) - self.bias) >= 1 
                
                if condition: 
                    
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights) # a -> learning rate 
                
                else: 
                    
                    self.weights  -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i , y_[index]))
                    self.bias -= self.learning_rate * y_[index] 
                    
            predicted = self.predict(X)
            accuracy = accuracy_score(y, predicted) 
            run["accuracy"].append(accuracy)
    
    def predict(self  , X): 
        approx = np.dot(X , self.weights) - self.bias
        
        return np.sign(approx)
        

#%% prepare dataset for the self made svm 

# Convert class names to numeric labels in y_train_selected and y_test_selected
y_train_selected_numeric = np.where(y_train_selected == class1, -1, 1)
y_test_selected_numeric = np.where(y_test_selected == class1, -1, 1)

# Print the shape of the filtered datasets
print("Shape of filtered training data:", X_train_selected.shape)
print("Shape of filtered training labels:", y_train_selected_numeric.shape)
print("Shape of filtered test data:", X_test_selected.shape)
print("Shape of filtered test labels:", y_test_selected_numeric.shape)

# we want to reshape the image from a 4D array to a 2D array
# This line extracts the number of samples, image height, image width, and number of channels (e.g., RGB channels) from the shape of the X_train array.
num_samples, img_height, img_width, num_channels = X_train_selected.shape

# it flattens the image data, converting each image into a one-dimensional vector.
# The resulting shape is (num_samples, img_height * img_width * num_channels),
X_train_selected = X_train_selected.reshape(num_samples, -1)
num_samples, img_height, img_width, num_channels = X_test_selected.shape
X_test_selected = X_test_selected.reshape(num_samples, -1)

#%% add the data to the self made svm 

# Define parameters
learning_rate = 0.01 #Model accuracy score with 0.001 and 0.01 200 itrs kernel and C=1.0 : 0.7710
lambda_param = 0.02 #Model accuracy score with polynomial kernel and C=1.0 : 0.8040
n_iters = 200
svm_self_made = SVM_from_scratch(learning_rate , lambda_param , n_iters) 


  
svm_self_made.fit(X_train_selected, y_train_selected_numeric)
predictions = svm_self_made.predict(X_test_selected)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test_selected_numeric, predictions)))

  
# Log metrics
accuracy = accuracy_score(y_test_selected_numeric, predictions)
run["accuracy"].append(accuracy)

#%% stop neptune 
run.stop()