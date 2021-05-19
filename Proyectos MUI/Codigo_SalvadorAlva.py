# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:35:45 2020

@author: L01506162
"""


# NLP texto para Salvador Alva 
#Natural languaje  Processing 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset= pd.read_csv("Salvador_2",delimiter = "\t",quoting=3, encoding='latin-1')
dataset2= pd.read_csv("Salvador_2_E",delimiter = "\t",quoting=3, encoding='latin-1')
#delimiter es para indicar que es TSV tabolador space, quoting =3 es para decir que ignoramos
# las comillas

#Limpieza de texto
"""import re 
import nltk
nltk.download('stopwords') #extraer plabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #es para volver todo infinitivo
corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #quitar signos puntuación 
    review = review.lower() #pasar a minúsculas todo
    review = review.split() #separar las palabras
    ps= PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review) #unir todo en palabras separdas por un espacio en blanco
    corpus.append(review)"""
X=dataset.iloc[:,:]

#Crear el Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 1500)
X= cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

"""
#Aplicar ejercicio de clasificación 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)"""
