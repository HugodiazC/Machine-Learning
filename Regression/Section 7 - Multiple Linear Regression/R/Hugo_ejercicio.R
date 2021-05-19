#Regresión linear múltiple
#Importar data set
dataset= read.csv('50_Startups.csv')
#Convertir las variables categóricas
dataset$State= factor(dataset$State,
                        levels = c("New York","California","Florida"),
                        labels= c(1,2,3))


#Dividir en áreas conjunto de trabajo y conjunto de test. 

library("caTools")
install.packages("cTools")
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio =0.8) #columna que se va a predecir "profit"
trainig_set = subset(dataset, split ==TRUE)
testing_set = subset(dataset, split ==FALSE)

#Ajustar el modelo de Regresión Linear Multiple con el Conjunto de Entrenamiento (training set)
regression = lm(formula = Profit ~ ., data= trainig_set)
#crea modelo, calcula profit en función de todas las otras variables "."

#Predecir los resultados con el conjunto de Testing
y_pred = predict(regression, newdata= testing_set)

#Construir un modelo óptimo con la eliminación hacia atrás
SL=0.05 #significance level
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
                data= dataset)
summary(regression)
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
                data= dataset)
summary(regression)
regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
                data= dataset)
summary(regression)
y_pred = predict(regression, newdata= testing_set)
