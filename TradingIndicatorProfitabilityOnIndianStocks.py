import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk("C:/Users/ankit/Desktop/Machine Learning/Projects/Algorithmic Trading/kaggle/input"): 
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        print(filename[:-21])
bnf_filepath = "C:/Users/ankit/Desktop/Machine Learning/Projects/Algorithmic Trading/kaggle/input/stock-market-data-nifty-100-stocks-5-min-data/NIFTY BANK_with_indicators_.csv"
dataset = pd.read_csv(bnf_filepath)
# dataset.index = pd.to_datetime(dataset['date'])
dataset.head()

date_col = pd.to_datetime([x[:-6] for x in dataset['date']])
dataset.index = date_col

dataset.head()

dataset.describe()

import plotly.graph_objects as go
from datetime import datetime

fig = go.Figure(data=[go.Candlestick(x=dataset.index[:5000],
                open=dataset['open'][:5000],
                high=dataset['high'][:5000],
                low=dataset['low'][:5000],
                close=dataset['close'][:5000])])

fig.show()

import seaborn as sns
sns.pairplot(dataset[['open', 'high', 'low', 'close']])

plt.figure(figsize=(12, 5))
plt.plot(dataset['close'][-5000:], label='Bank Nifty')
plt.plot(dataset['ema5'][-5000:], label='ema5')
plt.plot(dataset['ema20'][-5000:], label='ema20')
plt.title('Bank Nifty Close Price History')
plt.xlabel("2nd Jan 2015 - 20th Oct 2022 ")
plt.ylabel("Close Price INR")
plt.legend(loc="upper left")
plt.show()

def ema_5_20_crossover(data): # Expecting a dataframe, where close price, ema5 and ema20 value present
    buy_signal, sell_signal = [], []
    buy_price, sell_price = [], []
    flag = -1
    for i in range(len(data)):
        if data['ema5'][i] > data['ema20'][i]:
            if flag != 1:  # fresh buy signal
                buy_signal.append(1)
                buy_price.append(data['close'][i])
                sell_signal.append(np.nan)
                sell_price.append(np.nan)
                flag = 1
            else:  # if buy signal is already executed, then append everything as nan
                buy_signal.append(np.nan)
                buy_price.append(np.nan)
                sell_signal.append(np.nan)
                sell_price.append(np.nan)
        elif data['ema5'][i] < data['ema20'][i]:
            if flag != 0:  # sell signal
                sell_signal.append(0)
                sell_price.append(data['close'][i])
                buy_signal.append(np.nan)
                buy_price.append(np.nan)
                flag = 0
            else:   # if sell signal is already executed, then append everything as nan
                buy_signal.append(np.nan)
                buy_price.append(np.nan)
                sell_signal.append(np.nan)
                sell_price.append(np.nan)
        else:
            buy_signal.append(np.nan)
            buy_price.append(np.nan)
            sell_signal.append(np.nan)
            sell_price.append(np.nan)
    return buy_signal, buy_price, sell_signal, sell_price

buy_sell_signals = ema_5_20_crossover(dataset)

df_signals = pd.DataFrame(columns=['date', 'close', 'ema5', 'ema20', 'buy_signal', 'buy_price', 'sell_signal', 'sell_price'])
df_signals['date'] = dataset.index
df_signals['close'] = dataset['close'].values
df_signals['ema5'] = dataset['ema5'].values
df_signals['ema20'] = dataset['ema20'].values
df_signals['buy_signal'] = buy_sell_signals[0]
df_signals['buy_price'] = buy_sell_signals[1]
df_signals['sell_signal'] = buy_sell_signals[2]
df_signals['sell_price'] = buy_sell_signals[3]
df_signals.head(10)



plt.style.use('classic')
plt.figure(figsize=(16,8))
plt.plot(df_signals['close'][-3000:], label = 'Bank Nifty', alpha = 0.35)
plt.plot(df_signals['ema5'][-3000:], label = 'ema5', alpha = 0.35)
plt.plot(df_signals['ema20'][-3000:], label = 'ema20', alpha = 0.35)
plt.scatter(df_signals.index[-3000:], df_signals['buy_price'][-3000:], label ='Buy', marker='^',color='green')
plt.scatter(df_signals.index[-3000:], df_signals['sell_price'][-3000:],label='Sell', marker='v', color='red')
plt.title('Bank Nifty Buy-Sell Signals')
plt.xlabel('Time [in 5 minute]')
plt.ylabel('Bank Nifty price')
plt.legend(loc = 'upper left')
plt.show()

