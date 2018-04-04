load("features.RData")
library(dplyr)

a<-colnames(data)%in%feat_sele[,"var"]

data1=data[,a]


set.seed(562)
train<-sample(1:nrow(data),76000)
test<-(1:nrow(data))[-train]
testdata=data[test,]
traindata=data[train,]

##LASSO
library(glmnet)

mtx=model.matrix(fraud~.,data = data)[,-1]

model.lasso=glmnet(mtx[train,],traindata$fraud,alpha=1,family="binomial")

summary(model.lasso)

set.seed(562)
cv.out=cv.glmnet(mtx[train,],traindata$fraud,alpha=1,nfolds=5,family="binomial",type.measure = "class")
best.lambda=cv.out$lambda.min
best.lambda

par(mfrow=c(1,1))
plot(cv.out)

lasso.coef=predict(model.lasso,s=best.lambda,type="coefficients")
lasso.pred=predict(model.lasso,s=best.lambda,newx = mtx[test,],type = "class")
lasso.train=predict(model.lasso,s=best.lambda,newx = mtx[train,],type = "class")

lasso.pred.prob=predict(model.lasso,s=best.lambda,newx = mtx[test,],type = "response")
lasso.train.prob=predict(model.lasso,s=best.lambda,newx = mtx[train,],type = "response")

colnames(lasso.pred)[1]="fraud"
colnames(lasso.pred.prob)[1]="fraud"

lasso.pred.data%>%
  filter(fraud=="1")%>%
  summarise(n())

par(mfrow=c(1,1))
plot(model.lasso,xvar = "dev",label = TRUE)

par(mfrow=c(1,1))
plot(model.lasso)

lasso.coef

lasso.pred.data=data.frame(lasso.pred)
lasso.pred.data%>%
  filter(fraud=="1")

table(lasso.pred,testdata$fraud)
mean(lasso.pred==testdata$fraud)


lasso.train.data=data.frame(lasso.train)

table(lasso.train,traindata$fraud)
mean(lasso.train==traindata$fraud)


lasso.pred.prob.data=data.frame(lasso.pred.prob)
lasso.pred.prob.data%>%
  filter(fraud>=0.5)%>%
  summarise(n())
