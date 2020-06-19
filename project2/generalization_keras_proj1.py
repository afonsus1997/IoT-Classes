import numpy
import matplotlib.pyplot as plt
import pandas
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report, make_scorer, \
    precision_score, f1_score, plot_confusion_matrix
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# convert an array of values into a dataset matrix

def split_sequences(x, y, n_steps):
    X, Y = list(), list()
    for i in range(len(x)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(x):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = x[i:end_ix], y[end_ix-1]
        X.append(seq_x)
        Y.append(seq_y)
    return array(X), array(Y)


# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset
dataframe = pandas.read_csv('Lab6-7-8_IoTGatewayCrash.csv', decimal='.')
dataset = dataframe.values

test_data = pandas.read_csv('Proj2_IoTGatewayCrash_2.csv', decimal='.')


# dataframe['High_requests'] = numpy.where(dataframe['Requests']>=0.2, 1, 0)
# test_data['High_requests'] = numpy.where(test_data['Requests']>=0.2, 1, 0)


cols = [col for col in dataframe.columns if col not in ['Falha']]
data = dataframe[cols]
target = dataframe['Falha']
x_train = data
y_train = target

cols = [col for col in test_data.columns if col not in ['Falha']]
data = test_data[cols]
target = test_data['Falha']
x_test = data
y_test = target


oversample = RandomOverSampler(sampling_strategy='minority')
x_train, y_train = oversample.fit_resample(x_train, y_train)
print(x_train.values)
print(y_train.values)


n_steps = 23
n_features = 2


# convert into input/output
x_train_splits, y_train_splits = split_sequences(x_train.values, y_train.values, n_steps)
x_test_splits, y_test_splits = split_sequences(x_test.values, y_test.values, n_steps)
# x_val_splits, y_val_splits = split_sequences(x_val.values, y_val.values, n_steps)


y_train_splits_pred = []
y_test_splits_pred = []
# y_val_splits_pred = []



# summarize the data
# for i in range(len(x_train)):
#     print(x_train_splits[i], y_train_splits[i])

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(x_train_splits, y_train_splits, epochs=200, verbose=1)



# for i in range(len(x_train_splits)):
#     x_train_splits[i] = numpy.reshape(x_train_splits[i], (1, n_steps, n_features))
# x_train_splits = x_train_splits[0]
# print(x_train_splits)

# for i in range(len(x_train_splits)):
y_train_splits_pred = model.predict(x_train_splits, verbose=1)
y_train_splits_pred = y_train_splits_pred.astype(numpy.uint8)
for i in range(len(y_train_splits_pred)):
    y_train_splits_pred[i] = y_train_splits_pred[i][0]

y_test_splits_pred = model.predict(x_test_splits, verbose=1)
y_test_splits_pred = y_test_splits_pred.astype(numpy.uint8)
for i in range(len(y_test_splits_pred)):
    y_test_splits_pred[i] = y_test_splits_pred[i][0]

# print(y_train_splits)
# print(y_train_splits_pred)
print("========TRAIN=========")
print("Accuracy:")
print(accuracy_score(y_train_splits, y_train_splits_pred))
print("Precision:")
print(precision_score(y_train_splits, y_train_splits_pred))
print("Recall:")
print(recall_score(y_train_splits, y_train_splits_pred))
print("F-Measure:")
print(f1_score(y_train_splits, y_train_splits_pred))
cm = confusion_matrix(y_train_splits, y_train_splits_pred)
print(cm)
print(classification_report(y_train_splits, y_train_splits_pred))

# print(y_test_splits)
# print(y_test_splits_pred)
print("========TEST=========")
print("Accuracy:")
print(accuracy_score(y_test_splits, y_test_splits_pred))
print("Precision:")
print(precision_score(y_test_splits, y_test_splits_pred))
print("Recall:")
print(recall_score(y_test_splits, y_test_splits_pred))
print("F-Measure:")
print(f1_score(y_test_splits, y_test_splits_pred))
cm = confusion_matrix(y_test_splits, y_test_splits_pred)
print(cm)
print(classification_report(y_test_splits, y_test_splits_pred))



