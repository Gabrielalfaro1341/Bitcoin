import pandas as pd
from matplotlib import pyplot as plt
from modelo_red import modelo_mlp
import numpy as np
from datetime import timedelta,datetime



df=pd.read_csv('precio_bitcoin.csv')[-100:]
df.set_index('time',inplace=True)
df.drop(['time.1'],axis=1)


prediccion_train,prediccion_test,y_train,y_test,resultados=modelo_mlp(df,10,10,0.8)








df['time'] = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')


#visualizacion

fig,ax=plt.subplots(1,1)

ax.plot(df['time'],df['last'], label='prediction target',alpha=.7)
ax.plot(prediccion_train['time'],prediccion_train['Value_pre'], label='prediction target',alpha=.7)
plt.title('training result')
ax.plot(resultados['time'],resultados['Value_pre'], label='prediction target',alpha=.7)

ax.plot(prediccion_test['time'],prediccion_test['Value_pre'], label='prediction target',alpha=.7)






plt.show()