import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix
import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

'''from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K

from numpy import array
from numpy import hstack
#import keras_metrics as km
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''


df = pd.read_csv('Lab6-7-8_IoTGatewayCrash.csv')

################ a) ################
'''
data = df[['Requests','Load']]
target = df['Falha']

data_train,data_test,target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

clf = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=500, alpha=0.001, solver='adam')
clf.fit(data_train, target_train)

y_pred = clf.predict(data_train)
accuracy = accuracy_score(target_train, y_pred)
precision = precision_score(target_train, y_pred)
recall = recall_score(target_train, y_pred)
f1 = f1_score(target_train, y_pred)

print("MLPC Classifier accuracy score: ", accuracy)
print("MLPC Classifier precision score: ", precision)
print("MLPC Classifier recall score: ", recall)
print("MLPC Classifier F-1 score: ", f1)
print("\n\n\n")

y_pred = clf.predict(data_test)

accuracy = accuracy_score(target_test, y_pred)
precision = precision_score(target_test, y_pred)
recall = recall_score(target_test, y_pred)
f1 = f1_score(target_test, y_pred)


print("MLPC Classifier accuracy score: ", accuracy)
print("MLPC Classifier precision score: ", precision)
print("MLPC Classifier recall score: ", recall)
print("MLPC Classifier F-1 score: ", f1)

cm = confusion_matrix(target_test, y_pred)
sns.heatmap(cm, center=True,annot=True)
plt.show()

'''





################ b)_vanilla ################
'''
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

data = df[['Requests','Load']]
target = df['Falha']

data_train,data_test,target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 50)

data_train_requests = array(data_train['Requests'])
data_train_load = array(data_train['Load'])
target_train = array(target_train)

data_train_requests = data_train_requests.reshape((len(data_train_requests), 1))
data_train_load = data_train_load.reshape((len(data_train_load), 1))
target_train = target_train.reshape((len(target_train), 1))

dataset = hstack((data_train_requests, data_train_load, target_train))

data_test_requests = array(data_test['Requests'])
data_test_load = array(data_test['Load'])
target_test = array(target_test)


data_test_requests = data_test_requests.reshape((len(data_test_requests), 1))
data_test_load = data_test_load .reshape((len(data_test_load ), 1))
target_test = target_test.reshape((len(target_test), 1))

dataset_test = hstack((data_test_requests, data_test_load, target_test))
########
n_steps = 4

X, y = split_sequences(dataset, n_steps)
n_features = X.shape[2]

X2, y2 = split_sequences(dataset_test, n_steps)


# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',keras.metrics.Precision(), keras.metrics.Recall()])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',precision_m, recall_m, f1_m])
model.compile(optimizer='adam', loss='mse')
##########################################
model.fit(X, y, epochs=200, verbose=0)



y_train_splits_pred = model.predict(X2, verbose=1)
y_train_splits_pred = y_train_splits_pred.astype(numpy.uint8)
for i in range(len(y_train_splits_pred)):
    y_train_splits_pred[i] = y_train_splits_pred[i][0]

y_test_splits_pred = model.predict(X2, verbose=1)
y_test_splits_pred = y_test_splits_pred.astype(numpy.uint8)
for i in range(len(y_test_splits_pred)):
    y_test_splits_pred[i] = y_test_splits_pred[i][0]

print("========TEST=========")
print("Accuracy:")
print(accuracy_score(y2, y_test_splits_pred))
print("Precision:")
print(precision_score(y2, y_test_splits_pred))
print("Recall:")
print(recall_score(y2, y_test_splits_pred))
print("F-Measure:")
print(f1_score(y2, y_test_splits_pred, average=None))

cm = confusion_matrix(y2, y_test_splits_pred)
sns.heatmap(cm, center=True,annot=True)
plt.show()
'''

'''
loss_m, accuracy_m2, precision_m2, recall_m2, f1_score_m = model.evaluate(X2, y2, verbose=0)
y_pred = []
for i in range(len(X2)):

    x_input = array(X2[i])
    x_input = x_input.reshape((1, n_steps, n_features))
    y_pred_x = model.predict(x_input, verbose=0)
    y_pred.append(y_pred_x[0][0])

for i in range(len(y_pred)):
    y_pred[i] = y_pred[i] * (-1)
    if y_pred[i] >= 0.5:
        y_pred[i] = 1.
    else:
        y_pred[i] = 0.


print("Vanilla LSTM Classifier accuracy score: ", accuracy_m2)
print("Vanilla LSTM  Classifier precision score: ", precision_m2)
print("Vanilla LSTM  Classifier recall score: ", recall_m2)
print("Vanilla LSTM  Classifier F-1 score: ", f1_score_m)
'''


################ b)_stacked ################

'''
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

data = df[['Requests','Load']]
target = df['Falha']

data_train,data_test,target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

data_train_requests = array(data_train['Requests'])
data_train_load = array(data_train['Load'])
target_train = array(target_train)

data_train_requests = data_train_requests.reshape((len(data_train_requests), 1))
data_train_load = data_train_load .reshape((len(data_train_load ), 1))
target_train = target_train.reshape((len(target_train), 1))

dataset = hstack((data_train_requests, data_train_load, target_train))

data_test_requests = array(data_test['Requests'])
data_test_load = array(data_test['Load'])
target_test = array(target_test)


data_test_requests = data_test_requests.reshape((len(data_test_requests), 1))
data_test_load = data_test_load .reshape((len(data_test_load ), 1))
target_test = target_test.reshape((len(target_test), 1))

dataset_test = hstack((data_test_requests, data_test_load, target_test))
########
n_steps = 5

X, y = split_sequences(dataset, n_steps)
n_features = X.shape[2]

X2, y2 = split_sequences(dataset_test, n_steps)


# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
##########################################
model.fit(X, y, epochs=200, verbose=0)


y_train_splits_pred = model.predict(X2, verbose=1)
y_train_splits_pred = y_train_splits_pred.astype(numpy.uint8)
for i in range(len(y_train_splits_pred)):
    y_train_splits_pred[i] = y_train_splits_pred[i][0]

y_test_splits_pred = model.predict(X2, verbose=1)
y_test_splits_pred = y_test_splits_pred.astype(numpy.uint8)
for i in range(len(y_test_splits_pred)):
    y_test_splits_pred[i] = y_test_splits_pred[i][0]

print("========TEST=========")
print("Accuracy:")
print(accuracy_score(y2, y_test_splits_pred))
print("Precision:")
print(precision_score(y2, y_test_splits_pred))
print("Recall:")
print(recall_score(y2, y_test_splits_pred))
print("F-Measure:")
print(f1_score(y2, y_test_splits_pred, average=None))

cm = confusion_matrix(y2, y_test_splits_pred)
sns.heatmap(cm, center=True,annot=True)
plt.show()'''

###############################################################################
###################################### 4 ######################################
###################################### 4 ######################################
###################################### 4 ######################################
###############################################################################


'''data = df[['Load','Requests',]]
target = df['Falha']

data_shifted = data
data_shifted['Requests_1'] = data_shifted['Requests'].shift(1)
data_shifted['Requests_2'] = data_shifted['Requests'].shift(2)
data_shifted = data_shifted[['Load','Requests_1','Requests_2']]


ast_row = len(data_shifted)
data_shifted = data_shifted.iloc[2:]
ast_row = len(target)
target = target.iloc[2:]
data_train,data_test,target_train, target_test = train_test_split(data_shifted,target, test_size = 0.70, random_state = 10, shuffle = False)
'''

###### oversample ######

'''
oversample = RandomOverSampler(sampling_strategy='minority')
oversample = RandomOverSampler(sampling_strategy=0.5)
data_train_oversample, target_train_oversample= oversample.fit_resample(data_train, target_train)

clf = MLPClassifier(hidden_layer_sizes=(500,500,500), max_iter=500, alpha=0.001, solver='adam')
clf.fit(data_train_oversample, target_train_oversample)
'''
###### undersample ######
'''
undersample = RandomUnderSampler(sampling_strategy='majority')
undersample = RandomUnderSampler(sampling_strategy=0.5)
data_train_undersample, target_train_undersample= undersample.fit_resample(data_train, target_train)
clf = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=500, alpha=0.001, solver='adam')
clf.fit(data_train_undersample, target_train_undersample)

'''
###### combined ######
'''
oversample = RandomOverSampler(sampling_strategy='minority')
oversample = RandomOverSampler(sampling_strategy=0.1)
data_train_combined, target_train_combined = oversample.fit_resample(data_train, target_train)

undersample = RandomUnderSampler(sampling_strategy='majority')
undersample = RandomUnderSampler(sampling_strategy=0.5)
data_train_combined, target_train_combined = undersample.fit_resample(data_train_combined, target_train_combined)


print(Counter(target_train_combined))
clf = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=500, alpha=0.001, solver='adam')
clf.fit(data_train_combined, target_train_combined)
'''

###### SMOTE ######

'''
sm = SMOTE(random_state=42)
data_train_smote, target_train_smote = sm.fit_resample(data_train, target_train)
clf = MLPClassifier(hidden_layer_sizes=(500,500,500), max_iter=500, alpha=0.001, solver='adam')
clf.fit(data_train_smote, target_train_smote)

'''

'''y_pred = clf.predict(data_test)

accuracy = accuracy_score(target_test, y_pred)
precision = precision_score(target_test, y_pred)
recall = recall_score(target_test, y_pred)
f1 = f1_score(target_test, y_pred)

print("MLPC Classifier accuracy score: ", accuracy)
print("MLPC Classifier precision score: ", precision)
print("MLPC Classifier recall score: ", recall)
print("MLPC Classifier F-1 score: ", f1)

cm = confusion_matrix(target_test, y_pred)
sns.heatmap(cm, center=True,annot=True)
plt.show()
'''


###############################################################################
###################################### 5 ######################################
###################################### 5 ######################################
###################################### 5 ######################################
###############################################################################

data = df[['Load','Requests']]

'''data['Sum'] = data['Requests'].rolling(3).sum()
data = data.iloc[2:]
data.drop('Requests', axis=1, inplace=True)'''

target = df['Falha']
#target = target.iloc[2:]
data_shifted = data
data_shifted['Requests_1'] = data_shifted['Requests'].shift(1)
data_shifted['Requests_2'] = data_shifted['Requests'].shift(2)

data_shifted['Request_3'] = [1 if x> 0.2 else 0 for x in data_shifted['Requests']]

data_shifted = data_shifted[['Load','Requests_1','Requests_2','Request_3']]
target = target.iloc[2:]
data_shifted = data_shifted.iloc[2:]
data_train,data_test,target_train, target_test = train_test_split(data_shifted,target, test_size = 0.30, random_state = 10, shuffle = False)


clf = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=500, alpha=0.001, solver='adam')
#clf = MLPClassifier(max_iter=600, verbose=0,  alpha= 0.05, hidden_layer_sizes= (39, 161, 6), learning_rate= 'constant', solver= 'adam', activation= 'relu', tol=0.0000001)
clf.fit(data_train, target_train)

################ combined ##################
'''
oversample = RandomOverSampler(sampling_strategy='minority')
oversample = RandomOverSampler(sampling_strategy=0.2)
data_train_combined, target_train_combined = oversample.fit_resample(data_train, target_train)
# 0.2 / 0.5
undersample = RandomUnderSampler(sampling_strategy='majority')
undersample = RandomUnderSampler(sampling_strategy=0.6)
data_train_combined, target_train_combined = undersample.fit_resample(data_train_combined, target_train_combined)

clf = MLPClassifier(hidden_layer_sizes=(500,500,500), max_iter=500, alpha=0.001, solver='adam')
clf.fit(data_train_combined, target_train_combined)'''

###### SMOTE ######
'''
sm = SMOTE(random_state=42)
data_train_smote, target_train_smote = sm.fit_resample(data_train, target_train)
clf = MLPClassifier(hidden_layer_sizes=(500,500,500), max_iter=500, alpha=0.001, solver='adam')
clf.fit(data_train_smote, target_train_smote)
'''

y_pred = clf.predict(data_test)

accuracy = accuracy_score(target_test, y_pred)
precision = precision_score(target_test, y_pred)
recall = recall_score(target_test, y_pred)
f1 = f1_score(target_test, y_pred)

print("MLPC Classifier accuracy score: ", accuracy)
print("MLPC Classifier precision score: ", precision)
print("MLPC Classifier recall score: ", recall)
print("MLPC Classifier F-1 score: ", f1)

cm = confusion_matrix(target_test, y_pred)
sns.heatmap(cm, center=True,annot=True)
plt.show()
