# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:41:53 2020

@author: L01506162
"""


# NLP texto para Salvador Alva 
#Natural languaje  Processing 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset = pd.read_csv("salvador_English_Complete.csv",sep='delimiter' ,encoding='latin-1')

dataset_N = pd.read_csv("salvador_English_Complete_N.csv",sep='delimiter' ,encoding='latin-1')

import re 
import nltk
nltk.download('stopwords') #extraer plabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #es para volver todo infinitivo
corpus=[]
for i in range(0,477):
    review = re.sub('[^a-zA-Z]',' ', dataset_N['Comments'][i]) #quitar signos puntuación 
    review = review.lower() #pasar a minúsculas todo
    review = review.split() #separar las palabras
    ps= PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review) #unir todo en palabras separdas por un espacio en blanco
    corpus.append(review)

#Crear el Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X= cv.fit_transform(corpus).toarray()

from nltk.probability import FreqDist
Frec_corpus = FreqDist(corpus)
for word in corpus:
    Frec_corpus[word.lower()]+1
Frec_corpus
X2= cv.fit_transform(Frec_corpus).toarray()
    
    

