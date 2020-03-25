import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from datetime import datetime


df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', na_values=['.'], sep=r'\s*,\s*', delimiter=';')

# df = df.dropna() #without this, the rolling average is a mess because of the NaN values
print(df)

# df["Time (UTC)"] = df["Time (UTC)"].map(lambda x: datetime.strptime(str(x), '%Y.%m.%d %H:%M:%S'))
df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%Y.%m.%d %H:%M:%S')
x = df['Time (UTC)']
Open = df['Open']
High = df['High']
Low = df['Low']
Close = df['Close']
Volume = df['Volume ']
# df['DCOILBRENTEU'] = df['DCOILBRENTEU'].astype(float)
# y = df[['DCOILBRENTEU']]



plt.subplot(141)
plt.plot(x, Open)
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


