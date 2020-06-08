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

scaler = StandardScaler()
# data[['Elapsed Days In Sales Stage', 'Sales Stage Change Count', 'Total Days Identified Through Closing', 'Total Days Identified Through Qualified', 'Opportunity Amount USD', 'Client Size By Revenue', 'Client Size By Employee Count', 'Revenue From Client Past Two Years', 'Ratio Days Identified To Total Days', 'Ratio Days Validated To Total Days', 'Ratio Days Qualified To Total Days', 'Deal Size Category']] = scaler.fit_transform(data[['Elapsed Days In Sales Stage', 'Sales Stage Change Count', 'Total Days Identified Through Closing', 'Total Days Identified Through Qualified', 'Opportunity Amount USD', 'Client Size By Revenue', 'Client Size By Employee Count', 'Revenue From Client Past Two Years', 'Ratio Days Identified To Total Days', 'Ratio Days Validated To Total Days', 'Ratio Days Qualified To Total Days', 'Deal Size Category']])


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1)
x_test, (x_val), y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

mlp = MLPClassifier(max_iter=600, verbose=10,  alpha= 0.05, hidden_layer_sizes= (39, 161, 6), learning_rate= 'constant', solver= 'adam', activation= 'relu', tol=0.0000001)
mlp.fit(x_train, y_train)


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)
pred_val = mlp.predict(x_val)


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

print("========VALIDATION=========")
print("Accuracy:")
print(accuracy_score(y_val, pred_val))
print("Precision:")
print(precision_score(y_val, pred_val))
print("Recall:")
print(recall_score(y_val, pred_val))

cm = confusion_matrix(y_val, pred_val)
print(cm)
print(classification_report(y_val, pred_val))
df_cm = pd.DataFrame(cm, range(2), range(2))
sns.heatmap(df_cm)
plt.show()

