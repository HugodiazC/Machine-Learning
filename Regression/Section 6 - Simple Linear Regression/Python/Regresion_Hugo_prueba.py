# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:10:43 2020

@author: L01506162
"""

#Regresión lineal simple. 

#Importado de librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#División de datos en "X" (datos a procesar) y "y" (datos a predecir )
dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,1].values

#Dividir el data set en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=1/3, random_state= 0)

#Crear modelo de regresión linear simple con el conjunto de entrenamiento
#En esta parte ajustamos el "regresor" a las variables de entrenamiento.
from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred=regression.predict(X_test)

#Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color="red") # esto es para los valores de entreganiemto
plt.plot(X_train, regression.predict(X_train), color="blue") #esto es para la recta de regresión
plt.title("Sueldo vs. Años de Experiencia (Conjuno de Entrnamiento")
plt.xlabel("Años de Experiencia") 
plt.ylabel("Sueldo (en dolares")
plt.show()

#Visualizar los resultados de test
plt.scatter(X_test, y_test, color="red") # esto es para los valores de entreganiemto
plt.plot(X_train, regression.predict(X_train), color="blue") #esto es para la recta de regresión
plt.title("Sueldo vs. Años de Experiencia (Conjuno de Entrnamiento")
plt.xlabel("Años de Experiencia") 
plt.ylabel("Sueldo (en dolares")
plt.show()