import requests
import json
import time
import pandas as pd
from matplotlib import pyplot as plt

from pandas.plotting import autocorrelation_plot


def getBtc():
      response = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd")
      response = response.json()
      return response



contador=0

df=pd.DataFrame(columns=['high', 'last', 'timestamp', 'bid', 'vwap', 'volume', 'low', 'ask', 'open'])


validador=True
while validador==True:
    df=df.append(getBtc(),ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    df[['high', 'last', 'timestamp', 'bid', 'vwap', 'volume', 'low', 'ask', 'open']] = df[['high', 'last', 'timestamp', 'bid', 'vwap', 'volume', 'low', 'ask', 'open']].apply(pd.to_numeric)
    df['time']=pd.to_datetime(df['timestamp'],unit='s')
    df['last']=df['last'].astype(float)
    ax1.plot(df['time'][:],df['last'][:])

    if len(df)>1:

        df_diff = df.diff().dropna()
        df_diff.set_index(df['time'][1:],inplace=True)
        reversa_df_diff= df_diff.iloc[::-1]
        print(reversa_df_diff['last'])
        ax2.plot(df['time'][1:],df_diff['last']/df_diff['timestamp'])

        if len(df)>10 and len(df)%2==0:
            autocorrelation_plot(df_diff['last'])


        #if len(df_diff)>10:
            #resultado=modelo_arima(reversa_df_diff['last'])
            #print(resultado)


    contador+=1
    plt.show()

    time.sleep(10)







