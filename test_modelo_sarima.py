import pandas as pd
from matplotlib import pyplot as plt
from modelo_sarima import modelo_arima


df=pd.read_csv('precio_bitcoin.csv')[-100:]
df.set_index('time',inplace=True)
df.drop(['time.1'],axis=1,inplace=True)
print(df)
df_diff=df.diff().dropna()
prediccion,prediccion_futura=modelo_arima(df,120)
print(prediccion)
print(prediccion_futura)

fig,ax=plt.subplots(1,1)
df.index=pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

ax.plot(df.index,df['last'])
prediccion.predicted_mean.plot(ax=ax, label='prediccion', alpha=.7,style='--')
prediccion_futura.predicted_mean.plot(ax=ax, label='prediccion', alpha=.7, color='r',style='--')

plt.show()