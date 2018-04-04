
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, 
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv1 <- xgb.cv( params = params, data = dtrain, nrounds = 100,
                 nfold = 5, showsd = T, stratified = T, 
                 print.every.n = 10, early.stop.round = 20, maximize = F)
#67

min(xgbcv1$evaluation_log$test_error_mean)

xgb1 <- xgb.train(params = params, data = dtrain,
                   nrounds = 67, watchlist = list(val=dtest,train=dtrain),
                   print.every.n = 10,
                   early.stop.round = 10, maximize = F , eval_metric = "error")

xgbpred <- predict(xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.01,1,0)

#confusion matrix
library(caret)
confusionMatrix(xgbpred, labels_test)

#view variable importance plot
mat <- xgb.importance(colnames(x_train_tr), model = xgb1)
xgb.ggplot.importance(importance_matrix = mat[1:20]) 

#convert characters to factors
fact_col <- colnames(x_train_tr)[sapply(x_train_tr,is.character)]

for(i in fact_col) set(x_train_tr,j=i,value = factor(x_train_tr[[i]]))
for (i in fact_col) set(x_test_tr,j=i,value = factor(x_test_tr[[i]]))

#create tasks
library(mlr)
adsdsd <- as.data.frame(x_train_tr)
adsdsd <- cbind(adsdsd, labels)
colnames(adsdsd)[49] <- c("fraud")
adsdsd$fraud <- as.factor(adsdsd$fraud)
traintask <- makeClassifTask (data = adsdsd,target = "fraud")

adsdsd_test <- as.data.frame(x_test_tr)
adsdsd_test <- cbind(adsdsd_test, labels_test)
colnames(adsdsd_test)[49] <- c("fraud")
adsdsd_test$fraud <- as.factor(adsdsd_test$fraud)
testtask <- makeClassifTask (data = adsdsd_test,target = "fraud")


#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic",
                      eval_metric="error", nrounds=67L, eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")),
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask,
                     resampling = rdesc, measures = acc,
                     par.set = params, control = ctrl, show.info = T)
mytune$y 
mytune$x


#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask,threshold = 0.1)

confusionMatrix(xgpred$data$response,xgpred$data$truth)


adsdsd_all <- as.data.frame(x_all_tr)
adsdsd_all <- cbind(adsdsd_all, labels_all)
colnames(adsdsd_all)[49] <- c("fraud")
adsdsd_all$fraud <- as.factor(adsdsd_all$fraud)
alltask <- makeClassifTask (data = adsdsd_all,target = "fraud")

xgpred1 <- predict(xgmodel,alltask,threshold = 0.1)
summary(xgpred1)
xgpred1$threshold
xgpred1$data
confusionMatrix(xgpred1$data$response,xgpred1$data$truth)
