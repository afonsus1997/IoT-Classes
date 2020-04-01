import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from datetime import datetime


df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', na_values=['.'], sep=r'\s*,\s*', delimiter=';', decimal=',')
df.replace(0, np.nan, inplace=True)

df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%Y.%m.%d %H:%M:%S')
x = df['Time (UTC)']
Open = df['Open'].fillna(df['Open'].mean(skipna=True))
High = df['High'].fillna(df['High'].mean(skipna=True))
Low = df['Low'].fillna(df['Low'].mean(skipna=True))
Close = df['Close'].fillna(df['Close'].mean(skipna=True))
Volume = df['Volume '].fillna(df['Volume '].mean(skipna=True))
# df['DCOILBRENTEU'] = df['DCOILBRENTEU'].astype(float)
# y = df[['DCOILBRENTEU']]


def detect_outliers(x, y, k):
    avg = y.mean()
    std_dev = y.values.std(ddof=1)
    # print("Average: " + str(avg) + "\n")
    # print("Std dev: " + str(std_dev) + "\n")
    down = y[(y < avg - k * std_dev)]
    up = y[(y > avg + k * std_dev)]
    # print("Outliers below average-K*sigma\n")
    # print(down)
    # print("Outliers above average+K*sigma\n")
    # print(up)
    # print("All outliers \n")
    # all = all.concat
    # print(type(up.append(down).index.array))
    return up.append(down).index.array




    # print(std_dev)

plt.subplot(141)
outliers = detect_outliers(x, Open, 1)
Open_n = Open.drop(outliers)
x_n = x.drop(outliers)
print(Open)
print(Open_n)

print(type(x))
print(type(Open))
print(type(Open_n))


plt.plot(x_n, Open_n)
plt.subplot(142)
plt.plot(x, High)
plt.subplot(143)
plt.plot(x, Low)
plt.subplot(144)
plt.plot(x, Close)


# plt.subplot(142)
# plt.plot(x, y_minmax)
# plt.subplot(143)
# plt.plot(x, y_zscore)
# plt.subplot(144)
# plt.plot(x, y_avg)
# plt.gcf().autofmt_xdate()
#
plt.show()


