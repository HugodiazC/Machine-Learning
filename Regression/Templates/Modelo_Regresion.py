# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:05:32 2020

@author: L01506162
"""


#Plantilla regresión lineal: simple, múltiple, polinómica. 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
"""Ajustar siempre el dataset a descargar"""
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values #Columna a predecir

#Dividir el data set en conjunto de entrenamiento y conjunto de prueba
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state= 0)"""
  
#Escalado de variables
#Esto sirve para que las variables grandes queden en una escala numérica cercana
#por ejemplo una variable de 83000 al instante se transfrma en un no. de -1 a 1
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train) #Convertir nuestro X_train en escala de -1 a 1
X_test= sc_X.transform(X_test) #Convertir nuestro X_test en escala de -1 a 1


#Ajustar la regresión con el dataset


#Predicción de nuestros modelos
y_pred = regression.predict([[6.5]])

#Visualiación
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regression.predict([[6.5]]), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

