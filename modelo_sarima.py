from statsmodels.tsa.arima_model import ARIMA

def modelo_arima(df):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # instanciar modelo
    model=ARIMA(df,order=(2,1,2))
    # ajustar modelo
    results = model.fit()

    # mirar el AIC
    return results.summary()