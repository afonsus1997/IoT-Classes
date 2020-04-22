from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report, make_scorer, \
    precision_score, f1_score
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

def tuples_sum(nbval,total,order=True) :
    """
        Generate all the tuples L of nbval positive or nul integer
        such that sum(L)=total.
        The tuples may be ordered (decreasing order) or not
    """
    if nbval == 0 and total == 0 : yield tuple() ; raise StopIteration
    if nbval == 1 : yield (total,) ; raise StopIteration
    if total==0 : yield (0,)*nbval ; raise StopIteration
    for start in range(total,0,-1) :
        for qu in tuples_sum(nbval-1,total-start) :
            if qu[0]<=start :
                sol=(start,)+qu
                if order : yield sol
                else :
                    l=set()
                    for p in permutations(sol,len(sol)) :
                        if p not in l :
                            l.add(p)
                            yield p

import numpy


sales_data = pd.read_csv('WA_Fn-UseC_-Sales-Win-Loss.csv', decimal='.')

le = preprocessing.LabelEncoder()
#17 features
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

cols = [col for col in sales_data.columns if col not in ['Opportunity Number', 'Opportunity Result']]
data = sales_data[cols]
target = sales_data['Opportunity Result']



# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
# scaler.fit(x_test)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)
data[['Elapsed Days In Sales Stage', 'Sales Stage Change Count', 'Total Days Identified Through Closing', 'Total Days Identified Through Qualified', 'Opportunity Amount USD', 'Client Size By Revenue', 'Client Size By Employee Count', 'Revenue From Client Past Two Years', 'Ratio Days Identified To Total Days', 'Ratio Days Validated To Total Days', 'Ratio Days Qualified To Total Days', 'Deal Size Category']] = scaler.fit_transform(data[['Elapsed Days In Sales Stage', 'Sales Stage Change Count', 'Total Days Identified Through Closing', 'Total Days Identified Through Qualified', 'Opportunity Amount USD', 'Client Size By Revenue', 'Client Size By Employee Count', 'Revenue From Client Past Two Years', 'Ratio Days Identified To Total Days', 'Ratio Days Validated To Total Days', 'Ratio Days Qualified To Total Days', 'Deal Size Category']])
# print([col for col in cols if col not in ['Supplies Subgroup', 'Region', 'Route To Market', 'Opportunity Result', 'Competitor Type', 'Supplies Group']])
# print(data)
# daata = pd.DataFrame(data)
# daata.to_csv('F:\Documents\GitHub\IoT-Classes\lab5\export_dataframe.csv', index = False, header=True)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=21)

mlp = MLPClassifier(max_iter=300, verbose=1)
hidden_layers = []
# hidden_layers = [(10, 5), (100, 50), (50, 10, 2), (50, 25)]
sum_test = [10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30, 35, 40, 50, 100, 150, 200]
for i in range(len(sum_test)):
    hidden_layers += list(tuples_sum(1,sum_test[i],order=False)) + list(tuples_sum(2,sum_test[i],order=False)) + [(150, 100, 50)]
print(hidden_layers)

parameter_space = {
    'hidden_layer_sizes': hidden_layers,
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001, 0.05, 0.5, 1, 10],
    'learning_rate': ['constant'],
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}
skf = StratifiedKFold(n_splits=2)
clf = GridSearchCV(mlp, parameter_space, n_jobs=4, cv=skf)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)

# Best paramete set
# print('Best parameters found:\n', clf.best_params_)

# # All results
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

optimization_results = pd.DataFrame(clf.cv_results_)
optimization_results = optimization_results.sort_values(by='rank_test_score', ascending=False)
optimization_results.to_csv('F:\Documents\GitHub\IoT-Classes\lab5\optimization_results.csv', index = False, header=True)


# {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'constant', 'solver': 'sgd'}
#


#{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
# mlp = MLPClassifier(max_iter=300, verbose=10, activation= 'tanh', hidden_layer_sizes= (150, 100, 50), random_state=1)
# {'hidden_layer_sizes': (9, 3, 2), 'learning_rate': 'constant', 'solver': 'sgd'}



# mlp = MLPClassifier(max_iter=100, verbose=10, hidden_layer_sizes= (5, 5, 5,), learning_rate= 'constant', solver= 'lbfgs', tol=0.0000001)

# clf.fit(x_train, y_train)
# plt.plot(clf.best_loss_)
# plt.show()
pred_test = clf.predict(x_test)
pred_train = clf.predict(x_train)



print("========TRAIN=========")
print("Accuracy:")
print(accuracy_score(y_train, pred_train))
print("Precision:")
print(precision_score(y_train, pred_train))
print("Recall:")
print(recall_score(y_train, pred_train))

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

cm = confusion_matrix(y_test, pred_test)
print(cm)
print(classification_report(y_test, pred_test))
# sns.heatmap(cm, center=True)
# plt.show()
