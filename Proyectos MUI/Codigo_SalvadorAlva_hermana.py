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
dataset= pd.read_csv("Salvador_2.csv",delimiter = "\t",quoting=3, encoding='latin-1')
dataset2= pd.read_csv("Salvador_2_E.csv",delimiter = "\t",quoting=3, encoding='latin-1')
dataset3= pd.read_csv("Resumen_NLTK_2.csv", encoding='latin-1')
dataset4= pd.read_csv("Data.csv", encoding='latin-1')
#delimiter es para indicar que es TSV tabolador space, quoting =3 es para decir que ignoramos
# las comillas
X = dataset3.iloc[:, 1:-1].values
y = dataset3.iloc[:, -1].values

Grafica_FINAL=np.asarray(dataset3)
list1_FINAL = Grafica_FINAL.tolist()

plt.scatter(X, y, color = 'red')

plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()





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
X_2=dataset2.iloc[:,:]
S_text= """hug
action
front facing
to thank
students
opening
input
support for
learning
change
road
capacity
sweetie
clarity
to collaborate
share
committed
community
trust
shape
congruence
contribution
comply
dedicated
disruption
disruptor
having fun
example
push
Congratulations
enriching
delivery
equipment
essence
listen to me
effort
hope
eternally
excellence
success
experiences
Congratulations
we will celebrate
strong
future
Thank you
honor
paw print
human
important
tireless
amazing
innovation
innovative
inspire
together
work
legacy
Leader
leadership
leaders of tomorrow
achievements
moment
reasons
opportunity
pride
watershed
passion
think
will endure
possible
positive
Dear
recognition
present
reinvent you
respect
revolutionize
feel
talent
job
transformation
transcendence
only
value
brave
lifetime
vision"""

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
dataset_tokenize= word_tokenize(S_text)
len(dataset_tokenize)
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import ne_chunk
NE_tags= nltk.pos_tag(dataset_tokenize)
import nltk
nltk.download('maxent_ne_chunker')
import nltk
nltk.download('words')
NE_TAGS_CHUNK= ne_chunk(NE_tags)
Grafica_NLTK=np.asarray(NE_tags)
Grafica_NLTK2=np.array(NE_tags)

X = NE_tags.iloc[:, 1:-1].values
y = NE_tags.iloc[:, -1].values
from nltk.probability import FreqDist
NE_tags_F= FreqDist(NE_tags)
Grafica_Freq=np.asarray(NE_tags_F)
len(NE_tags)
NE_tags_F_array= FreqDist(Grafica_NLTK)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X= cv.fit_transform(NE_tags).toarray()
y=dataset.iloc[:,1].values
Grafica_lower = Grafica_NLTK.lower() 

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X= cv.fit_transform(Grafica_lower).toarray()

"""
import re 
import nltk
nltk.download('stopwords') #extraer plabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #es para volver todo infinitivo
corpus=[]
for i in range(0,100):
    review = re.sub('[^a-zA-Z]',' ', dataset2[0][i]) #quitar signos puntuación 
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
y=dataset.iloc[:,1].values





#Crear el Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 1500)
X= cv.fit_transform(corpus).toarray()



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
