from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import pandas as pd
from matplotlib import pyplot as plt
def modelo_arima(df,segundos):



    # instanciar modelo
    model=SARIMAX(df['last'],order=(1,0,0),trend='c',
                  enforce_stationarity=False,
                  simple_differencin=True)
    # ajustar modelo
    results = model.fit(disp=0)
    prediccion=results.get_prediction(start=df.index[1], dynamic=False)
    # mirar el AIC
    #prediccion futura
    new_datetime = timedelta(seconds=segundos
                             )
    new_datetime1= timedelta(seconds=segundos/12
                             )
    start_date = datetime.strptime(str(df.index[-1]), '%Y-%m-%d %H:%M:%S')
    tiempo_final=start_date+new_datetime
    print(tiempo_final)
    prediccion_futura=results.get_prediction(start=start_date,end=tiempo_final, dynamic=False)
    return prediccion,prediccion_futura


