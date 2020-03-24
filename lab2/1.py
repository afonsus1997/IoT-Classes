import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from datetime import datetime


df = pd.read_csv('DCOILBRENTEU.csv', na_values=['.'])
df = df.dropna() #without this, the rolling average is a mess because of the NaN values
print(df)

df['DATE'] = df['DATE'].map(lambda x: datetime.strptime(str(x), '%Y-%m-%d'))
x = df['DATE']
df['DCOILBRENTEU'] = df['DCOILBRENTEU'].astype(float)
y = df[['DCOILBRENTEU']]


print(df)

min_max_scaler = preprocessing.MinMaxScaler()
y_minmax = min_max_scaler.fit_transform(y)
print(y_minmax)
# y_norm=(y-y.min())/(y.max()-y.min())

z_scaler = StandardScaler()
z_scaler.fit(y)
y_zscore = z_scaler.transform(y)


y_avg = y.rolling(window=30).mean()


plt.subplot(141)
plt.plot(x, y)
plt.subplot(142)
plt.plot(x, y_minmax)
plt.subplot(143)
plt.plot(x, y_zscore)
plt.subplot(144)
plt.plot(x, y_avg)
plt.gcf().autofmt_xdate()

plt.show()


