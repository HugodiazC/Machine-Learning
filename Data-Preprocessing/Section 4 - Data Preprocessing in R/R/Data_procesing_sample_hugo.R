#preprosesado de datos
dataset= read.csv('Data.csv')
#tratamiento de valores NA
dataset$Age=ifelse(is.na(dataset$Age),
                   ave(dataset$Age,FUN = function(x) mean(x,na.rm = TRUE )) ,dataset$Age )
# dataset$Age// acá eligo la columna a realizar ejercicio de reemplazo
#=ifelse// expresión booleana,(is.na//acá tomo los NAN 
#(dataset$Age),#en casi de ser verdadero se ejecuta esto, #la segunda para falso )
#"ave" se encarga de producir un subconjunto de valores de x promediados
#FUN = function(x) mean(x,na.rm = TRUE )) el valor de x es reemplazado por la media
#de todos lo valores sin incluir los NA)
dataset$Salary=ifelse(is.na(dataset$Salary),
                   ave(dataset$Salary,FUN = function(x) mean(x,na.rm = TRUE )) ,dataset$Salary )

#Convertir las variables categóricas
dataset$Country= factor(dataset$Country,
                        levels = c("France","Spain","Germany"),
                        labels= c(1,2,3))

dataset$Purchased= factor(dataset$Purchased,
                        levels = c("No","Yes"),
                        labels= c(0,1))
#Dividir en áreas conjunto de trabajo y conjunto de test. 
install.packages("caTools")
library("caTools")
install.packages("cTools")
split = sample.split(dataset$Purchased, SplitRatio =0.8)
# espara dividirlo en 80%
trainig_set = subset(dataset, split ==TRUE)
testing_set = subset(dataset, split ==FALSE)

#Escalado de datos
trainig_set[,2:3]= scale(trainig_set[,2:3])
testing_set[,2:3]=scale(testing_set[,2:3])


