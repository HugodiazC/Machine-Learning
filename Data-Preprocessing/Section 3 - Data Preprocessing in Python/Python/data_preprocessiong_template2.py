# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

#Preprocesado instalar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("Data.csv")
#iloc sirve para localizar elementos por posición
X=dataset.iloc[:,:-1].values #primer : indica que quiere toda las columnas
#en el segundo estas indicando de la mis forma todas las filas :-1 menos la última
#.values significa que quiero extraer solo los valores del data frame
y=dataset.iloc[:,3].values #se elije la 3 por el conteo en python que empieza
#desde cero // minúscula se pone porque es vector, mayuscula por matriz

#Tratamiento de los NAN
from sklearn.impute import SimpleImputer #importar sklearn herramienta impute
imputer=SimpleImputer(missing_values=np.nan,strategy="mean",verbose=0) 
imputer=imputer.fit(X[:,1:3])  
X[:,1:3]=imputer.transform(X[:,1:3])
# la parte de X[:,1:3] fue para elegir solo las columnas 1 y 2
#recordando que en python el conteo empieza desde cero. 

#Codificar datos categóricos
##esto de refiere a los datos que son string
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,0]=le.fit_transform(X[:,0])

#esto es para convertir los datos en variables dummy 
#(asignar mismo valor)a todos los strings con one hot encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)
#Hacer la codificación pero ahora con los yes y No de la columna 3 
le_y = preprocessing.LabelEncoder()
y=le_y.fit_transform(y)

#Dividir el data set en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state= 0) 
#Se ponen 4 variables=(# variable de entrenamiento, variable a predecir, porcentaje de testeo, .2 y ramdom seed para que siempre obtener el mismo resultado)

#Escalado de variables
#Esto sirve para que las variables grandes queden en una escala numérica cercana
#por ejemplo una variable de 83000 al instante se transfrma en un no. de -1 a 1
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train) #Convertir nuestro X_train en escala de -1 a 1
X_test= sc_X.transform(X_test) #Convertir nuestro X_test en escala de -1 a 1


