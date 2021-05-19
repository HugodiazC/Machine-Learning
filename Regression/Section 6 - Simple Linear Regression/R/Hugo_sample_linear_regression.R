#Regresión linear simple

#Importar data set
dataset= read.csv('Salary_Data.csv')

#Dividir en áreas conjunto de trabajo y conjunto de test. 
install.packages("caTools")
library("caTools")
install.packages("cTools")
split = sample.split(dataset$Salary, SplitRatio =2/3)
# espara dividirlo en 80%
trainig_set = subset(dataset, split ==TRUE)
testing_set = subset(dataset, split ==FALSE)
 
#Ajustar el modelo de regresipon linear simple con el conjunto de entrenamiento
regressor= lm(formula =Salary~YearsExperience, data = trainig_set )

#Usar summary(regressor) para ver el desempeño de nuestro regresor lineal
#Predecir los resultados del conjunto de test
y_pred= predict(regressor, newdata = testing_set)

#Visualización de los datoe ne l conjunto de entreamiento
install.packages("tidyverse")

  
install.packages("ggplot2")
library("ggplot2")
ggplot()+
  geom_point(aes(x= trainig_set$YearsExperience , y= trainig_set$Salary ),
             colour = "red")+
  geom_line(aes(x = trainig_set$YearsExperience, 
                y = predict(regressor, newdata = trainig_set)), 
            colour = "blue")+
  ggtitle("Sueldo vs Años de Experiencia (conjunto de Entrenamiento)")+
  xlab("Años de Experiencia")+
  xlab("Sueldo en dolares")

ggplot()+
  geom_point(aes(x= testing_set$YearsExperience , y= testing_set$Salary ),
             colour = "red")+
  geom_line(aes(x = trainig_set$YearsExperience, 
                y = predict(regressor, newdata = trainig_set)), 
            colour = "blue")+
  ggtitle("Sueldo vs Años de Experiencia (conjunto de testing)")+
  xlab("Años de Experiencia")+
  xlab("Sueldo en dolares")
  
