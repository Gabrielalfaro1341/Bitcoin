import requests
import json
import time
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from modelo_sarima import modelo_arima
from modelo_red import modelo_mlp


def getBtc():
      response = requests.get("https://www.bitstamp.net/api/v2/ticker/btcusd")
      response = response.json()
      return response



contador=0

df=pd.DataFrame(columns=['high', 'last', 'timestamp', 'bid', 'vwap', 'volume', 'low', 'ask', 'open'])


validador=True
paso=False
while validador==True:
    df=df.append(getBtc(),ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    df[['high', 'last', 'timestamp', 'bid', 'vwap', 'volume', 'low', 'ask', 'open']] = df[['high', 'last', 'timestamp', 'bid', 'vwap', 'volume', 'low', 'ask', 'open']].apply(pd.to_numeric)
    df['time']=pd.to_datetime(df['timestamp'],unit='s')
    df['last']=df['last'].astype(float)
    df.set_index(df['time'],inplace=True)
    df = df.drop_duplicates()
    ax1.plot(df['time'][-100:],df['last'][-100:],label='Datos reales')
    print(df)


    if len(df)>1:

        df_diff = df.diff().dropna()
        df_diff.set_index(df['time'][1:],inplace=True)
        ax2.plot(df['time'][1:],df_diff['last']/df_diff['timestamp'],label='Pendiente')





        if len(df_diff)>100:
            paso=True
            df = df.asfreq(freq='20S', method='bfill')
            df.to_csv('precio_bitcoin_viernes_21_dic.csv')
            prediccion_arima, pre_arima_futura = modelo_arima(df[-100:], 120)
            pre_arima_futura.predicted_mean.plot(ax=ax1, label='prediccion_Sarimax', alpha=.7, color='r')
            prediccion_train, prediccion_test, y_train, y_test, resultados = modelo_mlp(df[-100:], 10, 6, 0.8)
            ax1.plot(resultados['time'], resultados['Value_pre'], label='Prediccion_red_neuronal', alpha=.7)
            ax1.legend()
            ax2.legend()
            plt.show()








    if not paso:
       time.sleep(20)
    else:
       time.sleep(16)







