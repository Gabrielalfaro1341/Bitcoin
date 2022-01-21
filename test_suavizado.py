from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from suavizado_minimos_locales import suavizar,minimos_locales,linea_tendencia_menor



df=pd.read_csv('precio_bitcoin.csv')
df.set_index('time',inplace=True)
df.drop(['time.1'],axis=1)


fig, ax = plt.subplots()
df['time'] = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

x, y = suavizar(df['time'], df['last'], 25)


tendenciax,tendenciay=minimos_locales(df['time'], df['last'],30)
linea_tendencia=linea_tendencia_menor(tendenciax,tendenciay,3)

ax.plot(df['time'],df['last'], label='prediction target',alpha=.7)
ax.plot(x, y)
ax.plot(tendenciax, tendenciay)
ax.plot(linea_tendencia['X_predict'],linea_tendencia['y_predict'])
print(tendenciax)
print(tendenciay)
plt.show()

linea_tendencia_menor(tendenciax,tendenciay,2)

