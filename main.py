import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

apiKey = 'key'

interval_var = '5min'
symbol = 'ETH'
path = 'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol='+symbol + \
    '&market=USD&interval='+interval_var+'&apikey=' + \
    apiKey+'&datatype=csv&outputsize=full'
df = pd.read_csv(path)
df = df[::-1].reset_index()
df = df.drop(['index'], axis=1)

# To find a slope of price line


def indSlope(series, n):
    array_sl = [j*0 for j in range(n-1)]
    for j in range(n, len(series)+1):
        y = series[j-n:j]
        x = np.array(range(n))
        x_sc = (x - x.min())/(x.max() - x.min())
        y_sc = (y - y.min())/(y.max() - y.min())
        x_sc = sm.add_constant(x_sc)
        model = sm.OLS(y_sc, x_sc)
        results = model.fit()
        array_sl.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(array_sl))))
    return np.array(slope_angle)

# True Range and Average True Range indicator


def indATR(source_DF, n):
    df = source_DF.copy()
    df['H-L'] = abs(df['high']-df['low'])
    df['H-PC'] = abs(df['high']-df['close'].shift(1))
    df['L-PC'] = abs(df['low']-df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df_temp = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df_temp

# generate data frame with all needed data


def PrepareDF(DF):
    ohlc = DF.iloc[:, [0, 1, 2, 3, 4, 5]]
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlc = ohlc.set_index('date')
    df = indATR(ohlc, 14).reset_index()
    df['slope'] = indSlope(df['close'], 5)
    df['channel_max'] = df['high'].rolling(10).max()
    df['channel_min'] = df['low'].rolling(10).min()
    df['position_in_channel'] = (
        df['close']-df['channel_min']) / (df['channel_max']-df['channel_min'])
    df = df.set_index('date')
    df = df.reset_index()
    return(df)

# find local mimimum / local maximum

# MIN


def isLCC(DF, i):
    df = DF.copy()
    LCC = 0

    if df['close'][i] <= df['close'][i+1] and df['close'][i] <= df['close'][i-1] and df['close'][i+1] > df['close'][i-1]:
        # dound low
        LCC = i-1
    return LCC

# MAX


def isHCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['close'][i] >= df['close'][i+1] and df['close'][i] >= df['close'][i-1] and df['close'][i+1] < df['close'][i-1]:
        # found high
        HCC = i
    return HCC


def getMaxMinChannel(DF, n):
    maxx = 0
    minn = 0
    for i in range(0, n-1):
        if maxx < DF['high'][len(DF)-i]:
            maxx = DF['high'][len(DF)-i]
        if minn > DF['low'][len(DF)-i]:
            minn = DF['low'][len(DF)-i]
    return(maxx, minn)


# convert time order
df = df[::-1]

prepared_df = PrepareDF(df)
lend = len(prepared_df)

# prepared_df[0:1000]['close'].plot()
# plt.show()
# prepared_df[0:1000]['slope'].plot()
# plt.show()
#prepared_df[0:1000][{'channel_max', 'channel_min', 'close'}].plot()
# plt.show()
# prepared_df[0:1000][{'position_in_channel'}].plot()
# plt.show()

prepared_df['hcc'] = [None] * lend
prepared_df['lcc'] = [None] * lend

for i in range(4, lend-1):
    if isHCC(prepared_df, i) > 0:
        prepared_df.at[i, 'hcc'] = prepared_df['close'][i]
    if isLCC(prepared_df, i) > 0:
        prepared_df.at[i, 'lcc'] = prepared_df['close'][i]


# backtesting

position = 0
deal = 0
eth_proffit_array = [[20, 1], [40, 1], [60, 2], [
    80, 2], [100, 2], [150, 1], [200, 1], [200, 0]]

prepared_df['deal_o'] = [None]*lend
prepared_df['deal_c'] = [None]*lend
prepared_df['earn'] = [None]*lend

for i in range(4, lend-1):
    prepared_df.at[i, 'earn'] = deal

    if position > 0:
        # long
        if(prepared_df['close'][i] < stop_price):
            # stop loss
            deal = deal-(open_price-prepared_df['close'][i])  # *position
            position = 0
            print('stop loss')
            prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]

        else:
            print(position, proffit_array)
            temp_arr = copy.copy(proffit_array)
            for j in range(0, len(temp_arr)-1):
                delta = temp_arr[j][0]
                contracts = temp_arr[j][1]
                if(prepared_df['close'][i] > (open_price+delta)):
                    print(i, position, open_price, prepared_df['close'][i])
                    print(proffit_array)
                    prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]
                    position = position-contracts
                    deal = deal+(prepared_df['close'][i]-open_price)*contracts
                    del proffit_array[0]

    elif position < 0:
        # short
        if(prepared_df['close'][i] > stop_price):
            # stop loss
            deal = deal-(prepared_df['close'][i]-open_price)  # *position
            position = 0
            prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]
            print('stop loss')
        else:
            temp_arr = copy.copy(proffit_array)
            for j in range(0, len(temp_arr)-1):
                delta = temp_arr[j][0]
                contracts = temp_arr[j][1]
                if(prepared_df['close'][i] < (open_price-delta)):
                    print(i, position, open_price, prepared_df['close'][i])
                    print(proffit_array)
                    prepared_df.at[i, 'deal_c'] = prepared_df['close'][i]
                    position = position+contracts
                    deal = deal+(open_price-prepared_df['close'][i])*contracts
                    del proffit_array[0]
    else:
        if(prepared_df['lcc'][i-1] != None):
            # long
            if(prepared_df['position_in_channel'][i-1] < 0.5):
                if((prepared_df['slope'][i-1]) < -5):
                    prepared_df.at[i, 'deal_o'] = prepared_df['close'][i]
                    print(i, 'open long position')
                    proffit_array = copy.copy(eth_proffit_array)
                    position = 10
                    open_price = prepared_df['close'][i]
                    stop_price = prepared_df['close'][i]*0.99
        if(prepared_df['hcc'][i-1] != None):
            # short
            if(prepared_df['position_in_channel'][i-1] > 0.5):
                if((prepared_df['slope'][i-1]) > 5):
                    prepared_df.at[i, 'deal_o'] = prepared_df['close'][i]
                    print(i, 'open short position')
                    proffit_array = copy.copy(eth_proffit_array)
                    position = -10
                    open_price = prepared_df['close'][i]
                    stop_price = prepared_df['close'][i]*1.01

hours = round(lend*5/60, 0)
print('Total erned in ', hours, ' hours =', int(deal), '$')
