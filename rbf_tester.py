# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:25:45 2024

@author: iason
"""
from keras import backend as K
from tensorflow.keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]

        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras.initializers import Initializer
from sklearn.cluster import KMeans


class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
from keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation

from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


#%%

# data = pd.read_csv('olive.csv',header=None)
# data.head(10) #Return 10 rows of data

# datatrans=np.transpose(data)
# print(datatrans[0].value_counts())
# datatrans[0].value_counts()[:].plot(kind='bar', alpha=0.5)
# plt.xlabel('\n Figure 1: Répartition selon classes \n', fontsize='17', horizontalalignment='center')
# plt.tick_params(axis='x',  direction='out', length=10, width=3)

# plt.show() #2300

# #data spliting
# X=data.iloc[2:570,:].values
# y = data.iloc[0:1,:].values
# #data rotation
# X=np.transpose(X)
# y=np.transpose(y)
# print('rotation ')
# print(X)
# print(y)
# #standarizing
# from sklearn.preprocessing import MinMaxScaler
# X = MinMaxScaler().fit_transform(X)
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y).toarray()
# print('resulats de scalling')
# print(X,y)

# from sklearn.model_selection import train_test_split
# from keras.optimizers import SGD
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)#80% train et 20% test

# # from sklearn.preprocessing import StandardScaler
# # sc_X = StandardScaler()
# # X_train = sc_X.fit_transform(X_train)
# # X_test = sc_X.transform(X_test)

# # sc_y = StandardScaler()
# # y_train = y_train.reshape((len(y_train), 1))
# # y_train = sc_y.fit_transform(y_train)
# # y_train = y_train.ravel()
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
 

#%%  data preperation for self-made svm 

class1, class2 = 1, 2  # You can choose the class indices 

# Filter training data and labelsa for the selected classes
selected_train_indices = np.where((y_train == class1) | (y_train == class2))[0]
X_train_selected = X_train[selected_train_indices]
y_train_selected = y_train[selected_train_indices]

# Filter test data and labels for the selected classes
selected_test_indices = np.where((y_test == class1) | (y_test == class2))[0]
X_test_selected = X_test[selected_test_indices]
y_test_selected = y_test[selected_test_indices]

#%% batching instead of 2 cateogires 

batch_size = 10000
X_batch = X_train[:batch_size]
y_batch = y_train[:batch_size]

batch_size_test = 2000
X_batch_test = X_test[:batch_size_test]
y_batch_test = y_test[:batch_size_test]

#%%
# we want to reshape the image from a 4D array to a 2D array
# This line extracts the number of samples, image height, image width, and number of channels (e.g., RGB channels) from the shape of the X_train array.
num_samples, img_height, img_width, num_channels = X_batch.shape

# it flattens the image data, converting each image into a one-dimensional vector.
# The resulting shape is (num_samples, img_height * img_width * num_channels),
X_batch = X_batch.reshape(num_samples, -1)
num_samples, img_height, img_width, num_channels = X_batch_test.shape
X_batch_test = X_batch_test.reshape(num_samples, -1)


#%% model initialization 
from keras.utils import to_categorical

input_shape = (32 * 32 * 3,)
num_classes = 10
num_classes_selected = 2 

model = Sequential()
rbflayer = RBFLayer(120,
                        initializer=InitCentersKMeans(X_batch),
                        betas=5.0,
                        input_shape=input_shape)
model.add(rbflayer)
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='mean_squared_error', 
                  optimizer='adam', metrics=['accuracy'])

y_train_one_hot = to_categorical(y_batch, num_classes)
y_test_one_hot = to_categorical(y_batch_test, num_classes)

#y_train_one_hot_selected = to_categorical(y_train_selected, num_classes_selected)
#y_test_one_hot_selected = to_categorical(y_test_selected, num_classes_selected)
#mean_squared_error
print(model.summary())
history1 = model.fit(X_batch, y_train_one_hot ,epochs=10, batch_size=128,)

import matplotlib.pyplot as plt
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['loss'])
plt.title('train accuracy and loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

# # saving to and loading from file
# z_model = f"Z_model.h5"
# print(f"Save model to file {z_model} ... ", end="")
# model.save(z_model)
# print("OK")

#model already saved in file

print("OK")

# Evaluate the model on the test data using `evaluate`






#%% 
# Convert class names to numeric labels in y_train_selected and y_test_selected
y_train_selected_numeric = np.where(y_train_selected == class1, -1, 1)
y_test_selected_numeric = np.where(y_test_selected == class1, -1, 1)