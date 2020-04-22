import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
    std_dev = y.values.std()
    print("Average: " + str(avg) + "\n")
    print("Std dev: " + str(std_dev) + "\n")
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


def delete_outliers(x, y, outlier_array):
    print(outlier_array)
    x_out = x.drop(outlier_array)
    y_out = y.drop(outlier_array)
    return [x_out, y_out]

def replace_previous(y, outlier_array):
    y_list = y.tolist()
    for i in range(len(outlier_array)):
        y_list[outlier_array[i]] = y_list[abs(outlier_array[i]-1)]
    return pd.Series(y_list)

def interpolate_outliers(x, y, outlier_array):
    y_local = y.copy(deep=True)
    for i in range(len(outlier_array)):
        y_local[outlier_array[i]] = np.nan
    temp = x.to_frame().join(y_local)
    print(temp)
    temp = temp.interpolate()
    print(temp)
    return temp['Volume ']

outliers = detect_outliers(x, Volume, 1)
[x_d, Volume_d] = delete_outliers(x, Volume, outliers)
Volume_p = replace_previous(Volume, outliers)
Volume_i = interpolate_outliers(x, Volume, outliers)

# fig, ax = plt.subplots(2, 2)
#
# ax[0, 0].plot(x, Volume)
# ax[0, 0].set_title('Original')
#
# ax[0, 1].plot(x_d, Volume_d)
# ax[0, 1].set_title('Deleted outliers')
#
# ax[1, 0].plot(x, Volume_p)
# ax[1, 0].set_title('Previous value')
#
# ax[1, 1].plot(x, Volume_i)
# ax[1, 1].set_title('Interpolated outliers')

def calc_id_day(df):

    y = df.diff(axis = 1, periods = -1)
    #print(y[ 0.02 > y['High'] > 0.01].count())
    #y = y.drop('Low', axis=1)
    #y.groupby(y.cut(y, np.arange(0.000,0.030,0.005))).count()
    return y['High']

plt.hist(calc_id_day(df[['High','Low']]))
#plt.hist(calc_dif(df['Close']))
plt.title('High to Low - Same day difference')
# plt.xticks(np.arange(0.000,0.055,0.005125))
# plt.yticks(np.arange(0,1400,50))
plt.tight_layout()
plt.grid()
plt.show()

# plt.subplot(142)
# plt.plot(x, y_minmax)
# plt.subplot(143)
# plt.plot(x, y_zscore)
# plt.subplot(144)
# plt.plot(x, y_avg)
# plt.gcf().autofmt_xdate()
#
plt.show()


