rm(list=ls())
options(scipen =999)
setwd("C:/Users/Moose/Desktop/ml2")

set.seed(999)
data= read.csv("train.csv",comment.char ='', header = TRUE)

library(anytime)

#Extract the dates
dates <- lapply(data[,9:12], function(x) anytime(x))
dates = as.data.frame(dates)

imp_data  = data[c(4,14)]
final_data  = cbind(imp_data,dates)


#Prepair Predictors on the basis of time diffrence
deadline_status = difftime(final_data $state_changed_at ,final_data $deadline , units = c("secs"))
deadline_status = as.data.frame(deadline_status)

lunch_created = difftime(final_data$launched_at ,final_data$created_at  , units = c("secs"))
lunch_created = as.data.frame(lunch_created)

 status_created = difftime(final_data $state_changed_at ,final_data$created_at , units = c("secs"))
 status_created  = as.data.frame(status_created)

 status_launched = difftime(final_data $state_changed_at ,final_data$launched_at , units = c("secs"))
 status_launched   = as.data.frame(status_launched )
 
 
 deadline_created = difftime(final_data $deadline, final_data $created_at , units = c("secs"))
 deadline_created = as.data.frame(deadline_created)


deadline_launched_at = difftime(final_data $deadline, final_data $launched_at , units = c("secs"))
deadline_launched_at = as.data.frame(deadline_launched_at)



final_data = cbind (deadline_status,lunch_created,status_created, status_launched ,
                    deadline_created
                    ,final_data[,1:2])
#final_data$final_status= as.factor(final_data$final_status)

demo =final_data

demo  = sapply(demo,function(x) as.numeric(x))
demo=as.data.frame(demo)


library(data.table)
library(stringr)

library(clusterSim)

#normalizasion to make the range similer to all predictors
norm_data = data.Normalization(demo, type = "n4", normalization = "column")

train = norm_data[sample(nrow(norm_data), 90000,replace = F),]
test = norm_data[!(1:nrow(norm_data)) %in% as.numeric(row.names(train)), ]

train = as.data.table(train)
test = as.data.table(test)

library(xgboost)
library(caret)

#Model BUilding

dtrain = xgb.DMatrix(data = as.matrix(train[,-c('final_status'),with=F]), label = train$final_status)
dtest = xgb.DMatrix(data = as.matrix(test[,-c('final_status'),with=F]), label = test$final_status)

params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=5, max_delta_step =0.5,
               min_child_weight=1, subsample=1, colsample_bytree=1)



#to select how many nrounds will be better (where test error is minimun))

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n= 10, early_stop_round = 20, maximize = F)



xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 41, watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "error")

xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)
confusionMatrix (xgbpred, test$final_status)

 xgb.importance(colnames(dtrain), model = xgb1) #69.27 #after1-69.47

 
remove(data,dates,norm_data,deadline_status,demo,final_data,imp_data,lunch_created,test,train)


#model deployment

data= read.csv("test.csv",comment.char ='', header = TRUE)


dates <- lapply(data[,9:12], function(x) anytime(x))
dates = as.data.frame(dates)

id = data[c(1)]
imp_data  = data[c(4)]
final_data  = cbind(imp_data,dates)

deadline_status = difftime(final_data $state_changed_at ,final_data $deadline , units = c("secs"))
deadline_status = as.data.frame(deadline_status)

lunch_created = difftime(final_data$launched_at ,final_data$created_at  , units = c("secs"))
lunch_created = as.data.frame(lunch_created)

status_created = difftime(final_data $state_changed_at ,final_data$created_at , units = c("secs"))
status_created  = as.data.frame(status_created)

status_launched = difftime(final_data $state_changed_at ,final_data$launched_at , units = c("secs"))
status_launched   = as.data.frame(status_launched )


deadline_created = difftime(final_data $deadline, final_data $created_at , units = c("secs"))
deadline_created = as.data.frame(deadline_created)


deadline_launched_at = difftime(final_data $deadline, final_data $launched_at , units = c("secs"))
deadline_launched_at = as.data.frame(deadline_launched_at)



final_data = cbind (lunch_created,deadline_status,
                    status_created,status_launched,deadline_created ,final_data[c(1)])

#final_data$final_status= as.factor(final_data$final_status)

demo =final_data

demo  = sapply(demo,function(x) as.numeric(x))
demo=as.data.frame(demo)

norm_data = data.Normalization(demo, type = "n4", normalization = "column")
 
norm_data = as.data.table(norm_data)
 #, label = test$final_status
 dtest = xgb.DMatrix(data = as.matrix(norm_data))
 
 xgbpred <- predict (xgb1,dtest)
 xgbpred <- ifelse (xgbpred > 0.5,1,0)

output = cbind(id,xgbpred)
names(output)[2]="final_status"

write.csv(output, "my_Submission.csv",row.names = FALSE) #0.65754

