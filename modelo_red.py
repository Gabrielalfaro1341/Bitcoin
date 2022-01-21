from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import timedelta,datetime
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,cross_validate


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back)]

        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

def grid_mlp(X, Y):
    estimator = MLPRegressor()

    param_grid = {'hidden_layer_sizes': [(50, 100, 50), (100, 1)],
                  'activation': ['relu', 'tanh', 'logistic'],
                  'learning_rate': ['constant', 'adaptive'],
                  'solver': ['adam']}

    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, Y)

    best_params = grid_result.best_params_
    print(best_params)



    return best_params

def modelo_mlp(df,look_back,numero_predicciones,fracc):
    #modelo



#Preparacion de datos
    scaler=MinMaxScaler()
    df['scale_last'] = scaler.fit_transform(df['last'].values.reshape(-1, 1))
    df['scale_timestamp'] = scaler.fit_transform(df['last'].values.reshape(-1, 1))

    x, y = create_dataset(df['scale_last'],look_back)




    size = int(len(x) * fracc)
    x_train, x_test = x[0:size], x[size:len(x)]
    y_train, y_test = y[0:size], y[size:len(x)]


    best_params=grid_mlp(x_train, y_train)


    #entrenamiento

    clf = MLPRegressor(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation=best_params["activation"],
        solver=best_params["solver"],
        random_state=17,
        max_iter=5000,
        learning_rate=best_params['learning_rate'],


    )

    clf.fit(x_train, y_train)
    train_mse = clf.predict(x_train)
    test_mse = clf.predict(x_test)

    #creando dataframe de predicciones

    time_train = np.array(df.index[look_back:int(len(x) * fracc) + look_back])
    value_train_real=np.array(df['last'][look_back:int(len(x) * fracc) + look_back])
    time_test = np.array(df.index[int(len(x) * fracc) + look_back:])
    value_test_real = np.array(df['last'][int(len(x) * fracc) + look_back:])

    dataframe_x_train = pd.DataFrame(zip(time_train,value_train_real,train_mse), columns=['time','Value_real' ,'Value_pre_sca'])
    dataframe_x_train['Value_pre']=scaler.inverse_transform(dataframe_x_train['Value_pre_sca'].values.reshape(-1, 1))
    dataframe_x_train['time'] = pd.to_datetime(dataframe_x_train['time'], format='%Y-%m-%d %H:%M:%S')

    dataframe_x_test = pd.DataFrame(zip(time_test,value_test_real,test_mse), columns=['time','Value_real','Value_pre_sca'])
    dataframe_x_test['Value_pre']=scaler.inverse_transform(dataframe_x_test['Value_pre_sca'].values.reshape(-1, 1))
    dataframe_x_test['time'] = pd.to_datetime(dataframe_x_test['time'], format='%Y-%m-%d %H:%M:%S')



    scoring = {
        'abs_error': 'neg_mean_absolute_error',
        'squared_error': 'neg_mean_squared_error',
        'r2': 'r2'}

    scores = cross_validate(clf, x_train, y_train,
                            cv=10, scoring=scoring,
                            return_train_score=True,
                            return_estimator=True)


    dato_agregar = np.delete(x_test[-1], 0)
    predecir = np.append(dato_agregar, test_mse[-1])
    lista_x=list()
    lista_x.append(predecir)
    resultados=list()
    lista_tiempo=list()

    #preficcion futura
    new_datetime = timedelta(seconds=20
                             )



    date_time_obj = dataframe_x_test.iloc[-1]['time']


    for i in range(0,numero_predicciones):
        date_time_obj += new_datetime
        one_prediction=clf.predict(lista_x[i].reshape(1,look_back))
        resultados.append(one_prediction[0])
        valor_anterior=np.delete(lista_x[i],0)
        valor_anterior=np.append(valor_anterior,one_prediction)
        lista_x.append(valor_anterior)

        lista_tiempo.append(date_time_obj)

    dataframe_prediccion = pd.DataFrame(zip(lista_tiempo, resultados), columns=['time', 'Value_pre_sca'])
    dataframe_prediccion['Value_pre']=scaler.inverse_transform(dataframe_prediccion['Value_pre_sca'].values.reshape(-1, 1))
    dataframe_prediccion['time'] = pd.to_datetime(dataframe_prediccion['time'], format='%Y-%m-%d %H:%M:%S')










    return dataframe_x_train,dataframe_x_test,y_train,y_test,dataframe_prediccion



