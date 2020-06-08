from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report, make_scorer, \
    precision_score, f1_score, plot_confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
# from yellowbrick.classifier import ClassificationReport
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from itertools import permutations
import numpy
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import math

def runMLP(layersizes, features):
    print("\n\n")
    print(layersizes)
    print(features)

    device_data = pd.read_csv('Lab6-7-8_IoTGatewayCrash.csv', decimal='.')

    #shifted data
    device_data['Requests1'] = device_data.Requests.shift(1)
    device_data['Requests2'] = device_data.Requests.shift(1)
    device_data = device_data.drop(device_data.index[[0,1]])
    # print(device_data)

    device_data['High_requests'] = numpy.where(device_data['Requests']>=0.2, 1, 0)
    # device_data['High_load'] = numpy.where(device_data['Load']>=0.53, 1, 0)

    #normal test
    cols = features
    # cols = [col for col in device_data.columns if col not in ['Falha']]
    data = device_data[cols]
    target = device_data['Falha']





    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.6, shuffle=False)
    x_test, (x_val), y_test, y_val = train_test_split(x_test, y_test, test_size=0.25, shuffle=False)


    # #oversampling
    oversample = RandomOverSampler(sampling_strategy='minority')
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    # print(x_train.values)
    # print(y_train.values)

    #under and oversampling
    # over = RandomOverSampler(sampling_strategy=0.5)
    # # fit and apply the transform
    # x_train, y_train = over.fit_resample(x_train, y_train)
    # # define undersampling strategy
    # under = RandomUnderSampler(sampling_strategy=0.5)
    # # fit and apply the transform
    # x_train, y_train = under.fit_resample(x_train, y_train)
    # print(x_train.values)
    # print(y_train.values)


    mlp = MLPClassifier(max_iter=600, verbose=0,  alpha= 0.05, hidden_layer_sizes= layersizes, learning_rate= 'constant', solver= 'adam', activation= 'relu', tol=0.0000001)
    mlp.fit(x_train, y_train)


    pred_train = mlp.predict(x_train)
    pred_test = mlp.predict(x_test)
    pred_val = mlp.predict(x_val)



    train_accuracy = accuracy_score(y_train, pred_train)
    train_precision = precision_score(y_train, pred_train)
    train_recall = recall_score(y_train, pred_train)
    train_FM = float(2 * train_precision * train_recall) / (train_precision + train_recall)

    # cm = confusion_matrix(y_train, pred_train)
    # print(cm)
    # print(classification_report(y_train, pred_train))

    test_accuracy = accuracy_score(y_test, pred_test)
    test_precision = precision_score(y_test, pred_test)
    test_recall = recall_score(y_test, pred_test)
    test_FM = float(2 * test_precision * test_recall) / (test_precision + test_recall)

    print("Train F-Measure:")
    print(train_FM)
    print("Test F-Measure:")
    print(test_FM)


    # cm = confusion_matrix(y_test, pred_test)
    # print(cm)
    # print(classification_report(y_test, pred_test))

    # print("========VALIDATION=========")
    # print("Accuracy:")
    # print(accuracy_score(y_val, pred_val))
    # print("Precision:")
    # print(precision_score(y_val, pred_val))
    # print("Recall:")
    # print(recall_score(y_val, pred_val))

    # cm = confusion_matrix(y_val, pred_val)
    # print(cm)
    # print(classification_report(y_val, pred_val))
    # print("F-Measure:")
    # print(f1_score(y_test, pred_test, average=None))
    if(math.isnan(test_FM)):
        return 0
    else:
        return test_FM

