from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report, make_scorer, \
    precision_score, f1_score, plot_confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ClassificationReport
import seaborn as sns
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


device_data = pd.read_csv('Lab6-7-8_IoTGatewayCrash.csv', decimal='.')
test_data = pd.read_csv('Proj2_IoTGatewayCrash_2.csv', decimal='.')

#shifted data
device_data['Requests'] = device_data.Requests.shift(1)
device_data['Requests2'] = device_data.Requests.shift(1)
device_data = device_data.drop(device_data.index[[0,1]])
print(device_data)

device_data['High_requests'] = numpy.where(device_data['Requests']>=0.2, 1, 0)
# device_data['High_load'] = numpy.where(device_data['Load']>=0.53, 1, 0)


#shifted data
test_data['Requests'] = test_data.Requests.shift(1)
test_data['Requests2'] = test_data.Requests.shift(1)
test_data = test_data.drop(test_data.index[[0,1]])
print(test_data)

test_data['High_requests'] = numpy.where(test_data['Requests']>=0.2, 1, 0)
# device_data['High_load'] = numpy.where(device_data['Load']>=0.53, 1, 0)

#normal test
cols = [col for col in device_data.columns if col not in ['Falha']]
data = device_data[cols]
target = device_data['Falha']
x_train = data
y_train = target
print(cols)

cols = [col for col in test_data.columns if col not in ['Falha']]
data = test_data[cols]
target = test_data['Falha']
x_test = data
y_test = target
#
# crash = device_data.loc[device_data['Falha'] == 1]
# no_crash = device_data.loc[device_data['Falha'] == 0]
# # print(device_data.loc[device_data['Falha'] == 1])
# # print(device_data.loc[device_data['Falha'] == 0])
#
# fig, ax = plt.subplots()
# # ax=fig.add_axes([0,0,1,1])
# ax.set_xlabel('Requests')
# ax.set_ylabel('Load')
# ax.scatter(crash['Requests'], crash['Load'], color='r', label = 'fail')
# ax.scatter(no_crash['Requests'], no_crash['Load'], color='b', label = 'no fail')
# ax.legend()
# ax.grid(True)
# plt.show()



# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=False)
# x_test, (x_val), y_test, y_val = train_test_split(x_test, y_test, test_size=0.25, shuffle=False)


# #oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
x_train, y_train = oversample.fit_resample(x_train, y_train)
print(x_train.values)
print(y_train.values)

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


mlp = MLPClassifier(max_iter=600, verbose=0,  alpha= 0.05, hidden_layer_sizes= (39, 161, 6), learning_rate= 'constant', solver= 'adam', activation= 'relu', tol=0.0000001)
mlp.fit(x_train, y_train)


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)



print("========TRAIN=========")
print("Accuracy:")
print(accuracy_score(y_train, pred_train))
print("Precision:")
print(precision_score(y_train, pred_train))
print("Recall:")
print(recall_score(y_train, pred_train))
print("F-Measure:")
print(f1_score(y_train, pred_train))

cm = confusion_matrix(y_train, pred_train)
print(cm)
print(classification_report(y_train, pred_train))


print("========TEST=========")
print("Accuracy:")
print(accuracy_score(y_test, pred_test))
print("Precision:")
print(precision_score(y_test, pred_test))
print("Recall:")
print(recall_score(y_test, pred_test))
print("F-Measure:")
print(f1_score(y_test, pred_test))


cm = confusion_matrix(y_test, pred_test)
print(cm)
print(classification_report(y_test, pred_test))



