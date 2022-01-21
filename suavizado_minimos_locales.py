import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from datetime import timedelta,datetime
from sklearn.linear_model import LinearRegression

def suavizar(x,y,puntos):
    listax=list()
    listay=list()
    for i in range(0,len(x)):
        if i%puntos==0 and i!=0:

            promedio=x[i-puntos:i].mean()

            listax.append(promedio)

            promedio2 = y[i - puntos:i].mean()

            listay.append(promedio2)


    return listax,listay


def minimos_locales(listax,listay,comparar):
    minimo_x=list()
    minimo_y=list()
    mayor = True
    for i in range(len(listax)):

        if i>=10 and i<=len(listax)-comparar:
            for n in range(1,comparar):

                if listay[i]<listay[i-n] and listay[i]<listay[i+n] :
                    mayor=True
                else:
                     mayor=False
                     break
            if mayor==True:
                minimo_x.append(listax[i])
                minimo_y.append(listay[i])
        if i>len(listax)-comparar:
            for n in range(1,comparar):
                if listay[i]<listay[i-n]:
                    mayor=True
                    if n>len(listax)-i:
                        if listay[i]<listay[i-n]:
                            mayor=True
                        else:
                             mayor=False
                             break




                else:
                     mayor=False
                     break
            if mayor==True:
                minimo_x.append(listax[i])
                minimo_y.append(listay[i])


    return  minimo_x,minimo_y



def linea_tendencia_menor(minimox,minimoy,puntos):
    df=pd.DataFrame(list(zip(minimox[-puntos:],minimoy[-puntos:])),columns=['X','Y'])
    df['timestamp'] = df.X.values.astype(np.int64) // 10 ** 9
    print(df)




    #creacion de linea a partir de los 2 ultimos minimos locales
    Y = np.asarray(df['Y'])
    X = df[['timestamp']]


    model = LinearRegression().fit(X, Y)


    #predecir a 20 datos

    new_datetime=timedelta(seconds=20)

    date_time_obj = minimox[-1]
    x_predict=list()

    for i in range(0, 100):
        date_time_obj += new_datetime
        x_predict.append(date_time_obj)

    df1 = pd.DataFrame(x_predict, columns=['X_predict'])
    df1['timestamp'] = df1.X_predict.values.astype(np.int64) // 10 ** 9
    print(df)
    print(df1)


    y_predict=model.predict(df1[["timestamp"]])
    df1['y_predict']=y_predict
    print(df1)


    return df1



