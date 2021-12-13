from statsmodels.tsa.statespace.sarimax import SARIMAX

def modelo_arima(df):


    # instanciar modelo
    model=SARIMAX(df,order=(1,0,0))
    # ajustar modelo
    results = model.fit(disp=0)
    prediccion=results.get_prediction(start=df.index[1], dynamic=False)
    # mirar el AIC
    return prediccion