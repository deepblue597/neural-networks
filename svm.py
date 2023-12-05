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
from neptune.version import version as neptune_client_version
project = neptune.init_project(project="jason-k/example-project-tensorflow-keras", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZTI4NWM4OC0wMDg2LTQ2YTItYmFmMi1iZGQ3MmZhN2U5MDkifQ==")

run = neptune.init_run(project='jason-k/example-project-tensorflow-keras',api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZTI4NWM4OC0wMDg2LTQ2YTItYmFmMi1iZGQ3MmZhN2U5MDkifQ==")

# project["general/brief"] = "/svm.py"
# project["general/data_analysis"].upload("data_analysis.ipynb")
# project["dataset/v0.1"].track_files("s3://datasets/images")
# project["dataset/latest"] = project["dataset/v0.1"].fetch()
#%% data to be imported 
data = 'C:/Users/IASON/neural-networks/pulsar_stars.csv'

df = pd.read_csv(data)

# view dimensions of dataset

print(df.shape)

# let's preview the dataset

df.head()

#%% data descriptions 
# view the column names of the dataframe

col_names = df.columns

col_names

# remove leading spaces from column names

df.columns = df.columns.str.strip()

# view column names again

df.columns

# rename column names

df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']

#%% check distributions 

# check distribution of target_class column

df['target_class'].value_counts()

# view summary of dataset

df.info()


#%% outliers in numerical variables 

# view summary statistics in numerical variables

round(df.describe(),2)

# draw boxplots to visualize outliers

plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df.boxplot(column='IP Mean')
fig.set_title('')
fig.set_ylabel('IP Mean')


plt.subplot(4, 2, 2)
fig = df.boxplot(column='IP Sd')
fig.set_title('')
fig.set_ylabel('IP Sd')


plt.subplot(4, 2, 3)
fig = df.boxplot(column='IP Kurtosis')
fig.set_title('')
fig.set_ylabel('IP Kurtosis')


plt.subplot(4, 2, 4)
fig = df.boxplot(column='IP Skewness')
fig.set_title('')
fig.set_ylabel('IP Skewness')


plt.subplot(4, 2, 5)
fig = df.boxplot(column='DM-SNR Mean')
fig.set_title('')
fig.set_ylabel('DM-SNR Mean')


plt.subplot(4, 2, 6)
fig = df.boxplot(column='DM-SNR Sd')
fig.set_title('')
fig.set_ylabel('DM-SNR Sd')


plt.subplot(4, 2, 7)
fig = df.boxplot(column='DM-SNR Kurtosis')
fig.set_title('')
fig.set_ylabel('DM-SNR Kurtosis')


plt.subplot(4, 2, 8)
fig = df.boxplot(column='DM-SNR Skewness')
fig.set_title('')
fig.set_ylabel('DM-SNR Skewness')

#%% check if we have normal or skewed distribution 

# plot histogram to check distribution


plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df['IP Mean'].hist(bins=20)
fig.set_xlabel('IP Mean')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 2)
fig = df['IP Sd'].hist(bins=20)
fig.set_xlabel('IP Sd')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 3)
fig = df['IP Kurtosis'].hist(bins=20)
fig.set_xlabel('IP Kurtosis')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 4)
fig = df['IP Skewness'].hist(bins=20)
fig.set_xlabel('IP Skewness')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 5)
fig = df['DM-SNR Mean'].hist(bins=20)
fig.set_xlabel('DM-SNR Mean')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 6)
fig = df['DM-SNR Sd'].hist(bins=20)
fig.set_xlabel('DM-SNR Sd')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 7)
fig = df['DM-SNR Kurtosis'].hist(bins=20)
fig.set_xlabel('DM-SNR Kurtosis')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 8)
fig = df['DM-SNR Skewness'].hist(bins=20)
fig.set_xlabel('DM-SNR Skewness')
fig.set_ylabel('Number of pulsar stars')

#%% declare future vector and target variable 

X = df.drop(['target_class'], axis=1)

y = df['target_class']


#%% seperate data into training and test set 

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% train and test shape 

# check the shape of X_train and X_test

X_train.shape, X_test.shape


#%% feature scaling 

cols = X_train.columns

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#%% 

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

X_train.describe()


#%%  Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.

# instantiate classifier with default hyperparameters
svc=SVC() 


# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% adjusting C 

# =============================================================================
# Effect of C:
# 
# A smaller value of C leads to a wider margin but allows more training points to be misclassified. 
# This results in a simpler model that may generalize better to unseen data. 
# On the other hand, a larger value of C makes the optimization prioritize classifying all training points correctly, 
# possibly leading to a narrower margin.
# =============================================================================

# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100.0) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% C = 1000

# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0) 


# fit classifier to training set
svc.fit(X_train,y_train)


# make predictions on test set
y_pred=svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred))) 

#%% Linear Kernel 

# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0) 


# fit classifier to training set
linear_svc.fit(X_train,y_train)


# make predictions on test set
y_pred_test=linear_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

#%% C = 100 kernel = Linear 

# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0) 


# fit classifier to training set
linear_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


#%% C = 1000 kernel = Linear 

# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0) 


# fit classifier to training set
linear_svc1000.fit(X_train, y_train)


# make predictions on test set
y_pred=linear_svc1000.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% train and test set accuracy 

y_pred_train = linear_svc.predict(X_train)

y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


#%% Null accuracy 


y_test.value_counts()

# check null accuracy score

null_accuracy = (3306/(3306+274))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


#%% Polynomial kernel 

# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0) 


# fit classifier to training set
poly_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=poly_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% C = 100 kernel = poly 

# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0) 


# fit classifier to training set
poly_svc100.fit(X_train, y_train)


# make predictions on test set
y_pred=poly_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% Sigmoid kernel 

# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 


# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#%% C = 100 kernel = sigmoid 

# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0) 


# fit classifier to training set
sigmoid_svc100.fit(X_train,y_train)


# make predictions on test set
y_pred=sigmoid_svc100.predict(X_test)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

#bad performance the sigmoid 

# =============================================================================
# We get maximum accuracy with rbf and linear kernel with C=100.0. and the accuracy is 0.9832. 
# Based on the above analysis we can conclude that our classification model accuracy is very good. 
# Our model is doing a very good job in terms of predicting the class labels.
# 
# But, this is not true. Here, we have an imbalanced dataset. The problem is that accuracy is an inadequate measure
#  for quantifying predictive performance in the imbalanced dataset problem.
# 
# So, we must explore alternative metrices that provide better guidance in selecting models. 
# In particular, we would like to know the underlying distribution of values and the type of errors our classifer is making.
# 
# One such metric to analyze the model performance in imbalanced classes problem is Confusion matrix.
# =============================================================================

#%% Confusion matrix 

# visualize confusion matrix with seaborn heatmap
cm = confusion_matrix(y_test, y_pred_test)
    
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


#%% Classification report 



print(classification_report(y_test, y_pred_test))


#%% ROC curve 

# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a Pulsar Star classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


#%% ROC AUC 

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred_test)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


#%% Cross validated ROC AUC 

# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(linear_svc, X_train, y_train, cv=10, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

#%% Stratified k-fold Cross Validation with shuffle split

# =============================================================================
# k-fold cross-validation is a very useful technique to evaluate model performance. 
# But, it fails here because we have a imbalnced dataset. So, in the case of imbalanced dataset, 
# I will use another technique to evaluate model performance. It is called stratified k-fold cross-validation.
# 
# In stratified k-fold cross-validation, we split the data such that the proportions 
# between classes are the same in each fold as they are in the whole dataset.
# 
# Moreover, I will shuffle the data before splitting because shuffling yields much better result.
# =============================================================================


from sklearn.model_selection import KFold


kfold=KFold(n_splits=5, shuffle=True, random_state=0)

# =============================================================================
# In this example, random_state=0 ensures that the randomization is consistent across runs. 
# You could use a different integer value for random_state to get a different randomization 
# while still maintaining reproducibility with that specific seed value.
# 
# Random Number Generation:
# 
# In certain algorithms or procedures, randomness is involved. 
# For example, when shuffling data or splitting it into folds, randomization is often used.
# Setting a Seed:
# 
# By setting the random_state parameter to a specific value (an integer), 
# you are essentially fixing the seed for the random number generator. 
# This means that the sequence of random numbers generated will be the same every time you run the code
# with the same random_state value.
# Reproducibility:
# 
# Setting a seed is crucial when you want your results to be reproducible. 
# It ensures that, even though there is randomness involved, you get the same randomization every time you run your code.
# 
# =============================================================================
linear_svc=SVC(kernel='linear')


linear_scores = cross_val_score(linear_svc, X, y, cv=kfold)

#%% Print scores
print('Stratified cross-validation scores with linear kernel:\n\n{}'.format(linear_scores))

#%% mean value 
# print average cross-validation score with linear kernel

print('Average stratified cross-validation score with linear kernel:{:.4f}'.format(linear_scores.mean()))


#%% Stratified k-Fold Cross Validation with shuffle split with rbf kernel

rbf_svc=SVC(kernel='rbf')


rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)

# print cross-validation scores with rbf kernel

print('Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))

# print average cross-validation score with rbf kernel

print('Average stratified cross-validation score with rbf kernel:{:.4f}'.format(rbf_scores.mean()))


#%%  Hyperparameter Optimization using GridSearch CV

# import GridSearchCV
from sklearn.model_selection import GridSearchCV


# import SVC classifier
from sklearn.svm import SVC


# instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
svc=SVC() 



# declare parameters for hyperparameter tuning
parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]} 
              ]




grid_search = GridSearchCV(estimator = svc,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


#%% Run the grid search 

grid_search.fit(X_train, y_train)

# =============================================================================
# The `gamma` parameter in both the Radial Basis Function (RBF) kernel and the Polynomial kernel of a Support Vector Machine (SVM) plays a similar role, controlling the influence of individual training samples on the model. However, the scale and interpretation of `gamma` can differ between these kernels.
# 
# In the RBF kernel:
# 
# - A smaller `gamma` means a larger similarity radius, resulting in smoother decision boundaries.
# - A larger `gamma` means a smaller similarity radius, leading to more complex and intricate decision boundaries.
#   
# In the Polynomial kernel:
# 
# - `gamma` is a scaling factor that influences the effect of higher-degree polynomials.
# - A smaller `gamma` results in a smoother decision boundary in the polynomial feature space.
# - A larger `gamma` makes the model focus more on the samples that are close to the decision boundary, 
# creating a more complex decision boundary.
# 
# The reason you might observe different ranges for `gamma` in the two cases is due to the nature of the kernels 
# and their influence on the model.
# 
# Regarding common values for `gamma`:
# 
# 1. **RBF Kernel:**
#    - Common values for `gamma` in the RBF kernel are often chosen from a range of 0.1 to 10, 
#    depending on the scale of the input features and the characteristics of the dataset.
#    It's typical to start with a small value and then increase it if the model needs a more complex decision boundary.
# 
# 2. **Polynomial Kernel:**
#    - In the Polynomial kernel, the impact of `gamma` can be different, and typical values might 
#    range from 0.01 to 1.0. Again, the appropriate value depends on the dataset. Smaller 
#    values may result in smoother decision boundaries, while larger values can lead to more intricate boundaries.
# 
# It's important to note that the optimal values for `gamma` can vary based on the specific 
# characteristics of your data. Experimentation and cross-validation are often used to find 
# the best hyperparameters for a given problem. The appropriate values may depend on the scale 
# of your features, the complexity of the underlying patterns, and the overall characteristics of your dataset.
# =============================================================================


#%% best model 

# examine the best model


# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))


# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

#%% 

# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))

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

#%%

# Select two classes (e.g., 'airplane' and 'automobile')
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

#%%
# make predictions on test set
y_pred=sigmoid_svc.predict(X_test_selected)


# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test_selected, y_pred)))

#Model accuracy score with sigmoid kernel and C=1.0 : 0.6925

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
#%% cm 

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
#%% neptune stop 
#run.stop()



#%%
grid_search.fit(X_train_selected, y_train_selected)

#%% best model 

# examine the best model


# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))


# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

#%% 

# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))

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

# Select two classes (e.g., 'airplane' and 'automobile')
class1, class2 = 0, 1  # You can choose the class indices based on the CIFAR-10 class names

# Filter training data and labels for the selected classes
selected_train_indices = np.where((y_train == class1) | (y_train == class2))[0]
X_train_selected = X_train[selected_train_indices]
y_train_selected = y_train[selected_train_indices]

# Filter test data and labels for the selected classes
selected_test_indices = np.where((y_test == class1) | (y_test == class2))[0]
X_test_selected = X_test[selected_test_indices]
y_test_selected = y_test[selected_test_indices]

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

#%%
run.stop()


