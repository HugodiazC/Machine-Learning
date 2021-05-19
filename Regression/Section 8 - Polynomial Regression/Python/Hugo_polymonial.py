# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:52:27 2020

@author: L01506162
"""


#Regresión polinómica 
#Preprocesado instalar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar Dataset 
dataset=pd.read_csv("Position_Salaries.csv")
#iloc sirve para localizar elementos por posición
X=dataset.iloc[:,1:2].values #Se uso [:,1:2] para convertir en matris, de dejarlos como estaba
# [:,1] hubiera sido un vector. [:,1:2] no toma en cuenta la columna 2
y=dataset.iloc[:,2].values #se elije la 3 por el conteo en python que empieza
#desde cero // minúscula se pone porque es vector, mayuscula por matriz

#Dividir el data set en conjunto de entrenamiento y conjunto de prueba
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state= 0)""" 
#Se ponen 4 variables=(# variable de entrenamiento, variable a predecir, porcentaje de testeo, .2 y ramdom seed para que siempre obtener el mismo resultado)

#Ajustar la regression linear del dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) 

# Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))