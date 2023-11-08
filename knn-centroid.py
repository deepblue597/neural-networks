#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:29:10 2023

@author: iasonas kakandris 

Course : Neural Networks 
Project: 1o imiparadoteo 
"""

import matplotlib.pyplot as plt  # for data visualization purposes
import seaborn as sns  # for data visualization% matplotlib inline
from sklearn.neighbors import KNeighborsClassifier  # knn classifier
from sklearn.metrics import accuracy_score  # to check the accuracy
from sklearn.metrics import confusion_matrix  # To check for TP , TN , FP , FN
# to make the confusion matrix into a heatmap
from sklearn.metrics import classification_report
# instead of only 1 test and 1 train data set
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid  # kmeans clustering
from tensorflow.keras.datasets import cifar10  # to import our data
import random

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# make the rgb value from 0 - 255 --> 0 - 1 ==> scaling
X_train, X_test = X_train / 255.0, X_test / 255.0

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Verify the dataset shape
print("Training images shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

# %%


# from 1 col mul rows --> 1 row mul cols
print(y_train)
y_train = y_train.reshape(-1,)
print(y_train)

# test to see if it works properly


def showImage(x, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(x[index])
    plt.xlabel(class_names[y[index]])


showImage(X_train, y_train, random.randint(0, 9))

# the train and test data
print(X_train.shape, X_test.shape)

# %%

# we want to reshape the image from a 4D array to a 2D array
# This line extracts the number of samples, image height, image width, and number of channels (e.g., RGB channels) from the shape of the X_train array.
num_samples, img_height, img_width, num_channels = X_train.shape

# it flattens the image data, converting each image into a one-dimensional vector.
# The resulting shape is (num_samples, img_height * img_width * num_channels),
X_train = X_train.reshape(num_samples, -1)
num_samples, img_height, img_width, num_channels = X_test.shape
X_test = X_test.reshape(num_samples, -1)

# %%

# instantiate the model
knn_3 = KNeighborsClassifier(n_neighbors=3)  # 3 neighboors

knn_1 = KNeighborsClassifier(n_neighbors=1)  # 1 neighboors

# fit the model to the training set
knn_3.fit(X_train, y_train)

knn_1.fit(X_train, y_train)

# Predict the variable from the X_test
y_pred_3 = knn_3.predict(X_test)

y_pred_1 = knn_1.predict(X_test)

print("3 neighboors pred\n", y_pred_3)

print("1 neighboor pred\n", y_pred_1)

# %%

# We check the accuracy of our score by copmaring the test values (correct one) with the predicted values

print('Model accuracy score for 1 neighboor: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_1)))
print('Model accuracy score for 3 neighboors: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_3)))

# %%

y_pred_train_1 = knn_1.predict(X_train)
y_pred_train_3 = knn_3.predict(X_train)

# %%
# we do the same with the training set to see the overall accuracy
print('Training-set accuracy score for 1 neighboor: {0:0.4f}'. format(
    accuracy_score(y_train, y_pred_train_1)))
print('Training-set accuracy score for 3 neighboors: {0:0.4f}'. format(
    accuracy_score(y_train, y_pred_train_3)))

# %%
# check null accuracy score
# Null accuracy is the accuracy that could be achieved by always predicting the most frequent class.

null_accuracy = (1000/10000)

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# %%

# Create the  confusion matrixes
cm_1 = confusion_matrix(y_test, y_pred_1)
heatmap_1 = sns.heatmap(cm_1, annot=True, fmt='d', cmap='YlGnBu',
                        xticklabels=class_names, yticklabels=class_names)
heatmap_1.set_title("confusion matrix for 1 neighboor")
plt.xlabel('Predicted')
plt.ylabel('Actual')
print(classification_report(y_test, y_pred_1))
plt.show()

cm_3 = confusion_matrix(y_test, y_pred_3)

heatmap_3 = sns.heatmap(cm_3, annot=True, fmt='d', cmap='YlGnBu',
                        xticklabels=class_names, yticklabels=class_names)
heatmap_3.set_title("confusion matrix for 3 neighboors")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_test, y_pred_3))

# =============================================================================
# Precision
# Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).
#
# So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.
# Check the predicted airplanes, sum them up and divide the actual airplanes (vertical line)
#
# Mathematically, precision can be defined as the ratio of TP to (TP + FP).
#
# Recall
# Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.
#
# Recall identifies the proportion of correctly predicted actual positives.
#
# Mathematically, recall can be given as the ratio of TP to (TP + FN).
# Check the times the predicted airplane, sum them up and divide the actual airplanes (horizontal line)
#
# f1-score
# f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.
# not so useful in multi-class classification
#
# Support
# Support is the actual number of occurrences of the class in our dataset.
#
# =============================================================================

# %%

# Applying 10-Fold Cross Validation
# instead of taking a test data and a training data, we use multiple test and train
# to check how the score changes with different data

scores = cross_val_score(knn_1, X_train, y_train, cv=10, scoring='accuracy')

print('Cross-validation scores with 1nn:{}'.format(scores))


# compute Average cross-validation score

print('Average cross-validation score with 1nn: {:.4f}'.format(scores.mean()))

scores_3 = cross_val_score(knn_3, X_train, y_train, cv=10, scoring='accuracy')

print('Cross-validation scores with 3nn:{}'.format(scores_3))


# compute Average cross-validation score

print(
    'Average cross-validation score with 3nn: {:.4f}'.format(scores_3.mean()))

# %%

# ----------------------- K means clustering ------------------------------------------


centroid_classifier = NearestCentroid()
centroid_classifier.fit(X_train, y_train)

y_pred_centroid = centroid_classifier.predict(X_test)

print("centroid classifier pred\n", y_pred_centroid)

print('Model accuracy score for centroid classifier: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_centroid)))

y_pred_train_centroid = centroid_classifier.predict(X_train)

print('Training-set accuracy score for centroid classifier: {0:0.4f}'. format(
    accuracy_score(y_train, y_pred_train_centroid)))

# %%

cm_centroid = confusion_matrix(y_test, y_pred_centroid)

heatmap_cent = sns.heatmap(cm_centroid, annot=True, fmt='d',
                           cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
heatmap_cent.set_title("confusion matrix for centroid")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_test, y_pred_centroid))

# %%
scores_centroid = cross_val_score(centroid_classifier, X_train, y_train, cv=10)

# Print the cross-validation scores

print("Cross-Validation Scores:", scores_centroid)
print(
    'Average cross-validation score with 1nn: {:.4f}'.format(scores_centroid.mean()))

# %%
# ----------------------- Results ----------------------------

print('Model accuracy score for 1 neighboor: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_1)))
print('Model accuracy score for 3 neighboors: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_3)))
print('Model accuracy score for centroid classifier: {0:0.4f}'. format(
    accuracy_score(y_test, y_pred_centroid)))
