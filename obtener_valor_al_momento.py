import requests
import json
import time
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from modelo_sarima import modelo_arima


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
    df.set_index(df['time'],inplace=True)
    df = df.drop_duplicates()
    ax1.plot(df['time'][:],df['last'][:])


    if len(df)>1:

        df_diff = df.diff().dropna()
        df_diff.set_index(df['time'][1:],inplace=True)
        ax2.plot(df['time'][1:],df_diff['last']/df_diff['timestamp'])





        if len(df_diff)>6:
            print(df)
            df=df.asfreq(freq='10S',method='bfill')
            print(df)
            prediccion=modelo_arima(df['last'])
            prediccion.predicted_mean.plot(ax=ax1, label='prediccion', alpha=.7)




    contador+=1
    plt.show()

    time.sleep(10)







