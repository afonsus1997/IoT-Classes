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
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import math
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# device_data = pd.read_csv('Lab6-7-8_IoTGatewayCrash.csv', decimal='.')
device_data = pd.read_csv('Proj2_IoTGatewayCrash_2.csv', decimal='.')



# shifted data
device_data['Requests1'] = device_data.Requests.shift(1)
device_data['Requests2'] = device_data.Requests.shift(1)
device_data = device_data.drop(device_data.index[[0, 1]])
# print(device_data)

device_data['High_requests'] = np.where(device_data['Requests'] >= 0.2, 1, 0)
device_data['High_load'] = np.where(device_data['Load'] >= 0.53, 1, 0)
device_data.loc[(device_data['High_requests'] == 1) & (device_data['High_load'] == 1), 'High_features'] = 1

device_data['Requests_mm'] = device_data['Requests'].rolling(5).sum()
device_data['Load_mm'] = device_data['Load'].rolling(5).sum()
device_data.fillna(0, inplace=True)

# cols = [col for col in device_data.columns if col not in ['Falha']]
cols = ['High_requests', 'Load', 'Requests_mm']
data = device_data[cols]
target = device_data['Falha']

# device_data.to_csv('fuzzy_test.csv')

fz_Load = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'fz_Load')
fz_requests_high = ctrl.Antecedent(np.arange(0, 2, 1), 'fz_requests_high')
# fz_requests_1 = ctrl.Antecedent(np.arrange(0, 1, 0.01), 'fz_requests1')
fz_requests_mm = ctrl.Antecedent(np.arange(0, 3.56, 0.01), 'fz_requests_mm')

fz_fail = ctrl.Consequent(np.arange(0, 2, 1), 'fz_fail')

#ranges
fz_Load['high'] = fuzz.trimf(fz_Load.universe, [0.55, 1, 1])
fz_Load['medium'] = fuzz.trimf(fz_Load.universe, [0.45, 0.5, 0.65])
fz_Load['low'] = fuzz.trimf(fz_Load.universe, [0, 0, 0.55])

fz_requests_high['low'] = fuzz.trimf(fz_requests_high.universe, [1, 1, 1])
# fz_requests_high['medium'] = fuzz.trimf(fz_requests_high.universe, [0, 0, 0])
fz_requests_high['high'] = fuzz.trimf(fz_requests_high.universe, [0, 0, 0])

fz_requests_mm['high'] = fuzz.trimf(fz_requests_mm.universe, [1.8, 2, 3.55])
fz_requests_mm['medium'] = fuzz.trimf(fz_requests_mm.universe, [1.25, 1.5, 1.85])
fz_requests_mm['low'] = fuzz.trimf(fz_requests_mm.universe, [0, 0, 1.3])

fz_fail['fail'] = fuzz.trimf(fz_fail.universe, [1, 1, 1])
fz_fail['no_fail'] = fuzz.trimf(fz_fail.universe, [0, 0, 0])


#rulez
rule1 = ctrl.Rule((fz_Load['high'] & fz_requests_mm['high']) & fz_requests_high['low'], fz_fail['fail'])
rule2 = ctrl.Rule((fz_Load['low'] & fz_requests_mm['low']) & fz_requests_high['low'], fz_fail['no_fail'])
rule3 = ctrl.Rule((fz_Load['medium'] & fz_requests_mm['medium']) & fz_requests_high['low'], fz_fail['no_fail'])
rule4 = ctrl.Rule((fz_Load['medium'] & fz_requests_mm['high']) & fz_requests_high['low'], fz_fail['no_fail'])
rule5 = ctrl.Rule((fz_Load['high'] & fz_requests_mm['medium']) & fz_requests_high['low'], fz_fail['no_fail'])
rule6 = ctrl.Rule((fz_Load['medium'] & fz_requests_mm['low']) & fz_requests_high['low'], fz_fail['no_fail'])
rule7 = ctrl.Rule((fz_Load['low'] & fz_requests_mm['medium']) & fz_requests_high['low'], fz_fail['no_fail'])
rule8 = ctrl.Rule((fz_Load['high'] & fz_requests_mm['low']) & fz_requests_high['low'], fz_fail['no_fail'])
rule9 = ctrl.Rule((fz_Load['low'] & fz_requests_mm['high']) & fz_requests_high['low'], fz_fail['no_fail'])
rule10 = ctrl.Rule((fz_Load['high'] & fz_requests_mm['high']) & fz_requests_high['high'], fz_fail['no_fail'])
rule11 = ctrl.Rule((fz_Load['low'] & fz_requests_mm['low']) & fz_requests_high['high'], fz_fail['no_fail'])
rule12 = ctrl.Rule((fz_Load['medium'] & fz_requests_mm['medium']) & fz_requests_high['high'], fz_fail['no_fail'])
rule13 = ctrl.Rule((fz_Load['medium'] & fz_requests_mm['high']) & fz_requests_high['high'], fz_fail['no_fail'])
rule14 = ctrl.Rule((fz_Load['high'] & fz_requests_mm['medium']) & fz_requests_high['high'], fz_fail['no_fail'])
rule15 = ctrl.Rule((fz_Load['medium'] & fz_requests_mm['low']) & fz_requests_high['high'], fz_fail['no_fail'])
rule16 = ctrl.Rule((fz_Load['low'] & fz_requests_mm['medium']) & fz_requests_high['high'], fz_fail['no_fail'])
rule17 = ctrl.Rule((fz_Load['high'] & fz_requests_mm['low']) & fz_requests_high['high'], fz_fail['no_fail'])
rule18 = ctrl.Rule((fz_Load['low'] & fz_requests_mm['high']) & fz_requests_high['high'], fz_fail['no_fail'])

fuzzy_controller = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5,rule6,rule7,rule8,rule9, rule10, rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18])
fuzzy_controller_fail = ctrl.ControlSystemSimulation(fuzzy_controller,flush_after_run=50* 50+ 1)

output = []

for i in range(device_data.shape[0]):
    fuzzy_controller_fail.input['fz_Load'] = device_data.iloc[i]['Load']
    fuzzy_controller_fail.input['fz_requests_high'] = device_data.iloc[i]['High_requests']
    fuzzy_controller_fail.input['fz_requests_mm'] = device_data.iloc[i]['Requests_mm']

    fuzzy_controller_fail.compute()
    output.append(fuzzy_controller_fail.output['fz_fail'])

for i in range(len(output)):
    if output[i] >= 0.5:
        output[i] = 1
    else:
        output[i] = 0

print("========RESULTS=========")
print("Accuracy:")
print(accuracy_score(target, output))
print("Precision:")
print(precision_score(target, output))
print("Recall:")
print(recall_score(target, output))
print("F-Measure:")
print(f1_score(target, output))

cm = confusion_matrix(target, output)
print(cm)
print(classification_report(target, output))