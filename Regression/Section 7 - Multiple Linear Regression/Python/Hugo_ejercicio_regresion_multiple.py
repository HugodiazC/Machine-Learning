# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:25:38 2020

@author: L01506162
"""

#Regresión Linear Multiple
#Importado de librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#División de datos en "X" (datos a procesar) y "y" (datos a predecir )
dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,4].values

#Codificar datos categóricos
##esto de refiere a los datos que son string
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,3]=le.fit_transform(X[:,3])

#esto es para convertir los datos en variables dummy 
#(asignar mismo valor)a todos los strings con one hot encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)

X=X[:,1:]

#Dividir el data set en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state= 0)

#Ajustar el Modelo de Regresión Lineal Multiple con el Conjunto de Entrenamiento

from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(X_train, y_train)

#Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

#contruir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
#Para utilizar la librería de Statsmodels, se requiere la columna de 1 que se agregará abajo
#El código de abajo es para agregar una fila de 1 al principio de x 
X= np.append(arr= np.ones((50,1)).astype(int),values = X, axis=1) 
#axis=1 es para que se añada en columna // axis=0 sería para agregar en fila
SL=0.05
#crear variable que se quede con las variables que realmente tienen impacto
#en el resultado final de la regresión
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X).fit()
regressor_OLS.summary()

#despúes de analiar y ver que x2 al tener un p valor de .990 es la más alta
#la eliminamos y volvemos a lanzar al regresión lineal 
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y,exog=X).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS_1 = sm.OLS(endog=y,exog=X).fit()
regressor_OLS_1.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS_1 = sm.OLS(endog=y,exog=X).fit()
regressor_OLS_1.summary()



"""#Visualizar los resultados de test
plt.scatter(X_test, y_test, color="red") # esto es para los valores de entreganiemto
plt.plot(X_train, regression.predict(X_train), color="blue") #esto es para la recta de regresión
plt.title("Sueldo vs. Años de Experiencia (Conjuno de Entrnamiento")
plt.xlabel("Años de Experiencia") 
plt.ylabel("Sueldo (en dolares")
plt.show()"""
