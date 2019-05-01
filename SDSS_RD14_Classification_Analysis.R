rm(list=ls())
setwd("C:/Users/sean_/Documents/Graduate/STATS/Project")

library(rpart)
library(rattle)
library(ggplot2)
library(randomForest)
library(MASS)
library(tree)
library(gbm)
library(e1071)
library(cluster)
library(mclust)
library(nnet)
library(class)
library(tabplot)

####Classifcation Tree#####
data <- read.csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv", header = TRUE)
tableplot(data)
data <- data[-c(1,10)]
tree <- rpart(class ~ ., data = data, method = "class", control=rpart.control(minsplit=10, cp=0.001))
fancyRpartPlot(tree, main = "Classification Tree for the Zoo Data")

#### bagging####
set.seed(1)
train = sample (1: nrow(data ), nrow(data )/2)
data.test=data[-train,"class"]

bag.data=randomForest(as.factor(class)~.,data=data,subset=train,mtry=16,importance=TRUE)
bag.data

yhat.bag = predict(bag.data,newdata=data[-train,])
mean((as.character(yhat.bag))!=data.test)

plot(yhat.bag,data.test)

importance(bag.data)

varImpPlot(bag.data)

#### random forest####
set.seed(1)
rf.data=randomForest(as.factor(class)~.,data=data, type = classification,subset=train,mtry=4,importance=TRUE,ntree=500)
rf.data
yhat.rf = predict(rf.data,newdata=data[-train,])
mean(((as.character(yhat.rf))!=data.test))

importance(rf.data)

varImpPlot(rf.data)

#### Boosting ###
set.seed(1)
boost.data=gbm(as.factor(class)~.,data=data[-train,],distribution="tdist",n.trees=5000,interaction.depth=2, shrinkage = 0.001)
summary(boost.data)

yhat.boost=predict(boost.data,newdata=data[-train,],n.trees=5000)

data.test=data[-train,"class"]
levels(data.test)[3] <- '3' ##convert levels to numeric
levels(data.test)[2] <- '2'
levels(data.test)[1] <- '1'
mean(((round(yhat.boost))!=data.test))

## Hclustering
data <- read.csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv", header = TRUE)
X<-scale(data[,-c(1,10,14)])
hc <- hclust(dist(X), "ward.D")
plot(hc)
l4<-cutree(hc,3)
table(data[,14],l4)

## K means
data_kmeans<-kmeans(X,3)

plot(X, col = data_kmeans$cluster)

table(data[,14],data_kmeans$cluster)

library(cluster)
clusplot(X, data_kmeans$cluster, main='2D representation of the Cluster solution',
         color=TRUE, shade=TRUE,
         labels=2, lines=0)


## Model Based Clustering

mod1 <- Mclust(X,3)
summary(mod1, parameters = TRUE)
plot(mod1, what = "BIC")
table(data[,14], mod1$classification)


##Mixed Discriminant Analysis

data_delete<-rep(0,dim(X)[1])

k<-1
for(i in 1:dim(X)[1]){
  if(i%%4==0){data_delete[k]<-i; k<-k+1}
}
dataMclustDA <- MclustDA(X[-data_delete,], data[-data_delete,14])
summary(dataMclustDA, parameters = TRUE)
summary(dataMclustDA, newdata = X[data_delete,], newclass = data[data_delete,14])
plot(dataMclustDA, what = 'classification')

## 5 fold CV for neural network 
set.seed(1)
train<-sample(1:10000,8000)


data_nnet1 = tune.nnet(class~., data = data[train,], size = 1:30,tunecontrol = tune.control(sampling = "cross",cross=5))
summary(data_nnet1)
plot(data_nnet1)

cls<-class.ind(data[,14])
nn_data<-nnet(X[train,], cls[train,], size=25, decay=0, softmax=TRUE)
nn_pred<-predict(nn_iris, X[-train,], type="class")
table(data[-train,14],nn_pred)
classification_error <- 1- sum(nn_pred == data[-train,14])/length(nn_pred)
