rm(list=ls())
options(scipen =999)
setwd("C:/Users/Moose/Desktop/hackerearth")

set.seed(999)
library(data.table)
df = fread("train.csv", header = T , stringsAsFactors = FALSE, check.names = FALSE,na.strings=c("","NA"))


library(anytime)

#Extract the dates
dates <- lapply(df[,c("deadline","state_changed_at","created_at","launched_at")], function(x) anytime(x))
dates = as.data.frame(dates)

imp_data  = df[,c("goal","disable_communication","country","currency","backers_count" ,"final_status")]
final_data  = cbind(dates,imp_data)
remove(imp_data,dates)


#Prepair Predictors on the basis of time diffrence
deadline_status = difftime(final_data$state_changed_at,final_data $deadline , units = c("secs"))
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
                    deadline_created,deadline_launched_at
                    ,final_data[,c("goal","disable_communication","country","currency" ,"final_status" )])

remove(deadline_status,lunch_created,status_created, status_launched ,
       deadline_created,deadline_launched_at)


final_data$deadline_status = as.numeric(final_data$deadline_status)
final_data$lunch_created = as.numeric(final_data$lunch_created)
final_data$status_created = as.numeric(final_data$status_created)
final_data$status_launched = as.numeric(final_data$status_launched)
final_data$deadline_created = as.numeric(final_data$deadline_created)
final_data$deadline_launched_at = as.numeric(final_data$deadline_launched_at)
final_data$final_status = as.numeric(final_data$final_status)
final_data$disable_communication = as.factor(final_data$disable_communication)
final_data$country = as.factor(final_data$country)
final_data$currency = as.factor(final_data$currency)


final_data =  model.matrix(~.+0,data = final_data)

final_data = as.data.frame(final_data)

library(clusterSim)


library(xgboost)
library(data.table)



labels <- final_data$final_status

#removeing target variable from training dataset because it is already taken as labels
final_data =  final_data[,-c(28)]


final_data=as.matrix(final_data)

#preparing dataset to XGboost compatible dataset as it takes only matrix form
dtrain <- xgb.DMatrix(data = final_data,label = labels) 
remove(df,final_data)

#setting xgboost parameteres
params <- list(booster = "gbtree", objective = "reg:logistic", 
               eta=0.5, max_depth=20)


xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 10, nfold = 5, 
                 showsd = T, stratified = T, print_every_n= 10, early_stop_round = 20, maximize = F)


set.seed(123)

#model building
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 281, 
                   print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "error")


summary(xgb1)













df = fread("test.csv", header = T , stringsAsFactors = FALSE, check.names = FALSE,na.strings=c("","NA"))
project_id = df[,c("project_id")]

#Extract the dates
dates <- lapply(df[,c("deadline","state_changed_at","created_at","launched_at")], function(x) anytime(x))
dates = as.data.frame(dates)

imp_data  = df[,c("goal","disable_communication","country","currency")]
final_data_test  = cbind(dates,imp_data)
remove(imp_data,dates)


#Prepair Predictors on the basis of time diffrence
deadline_status = difftime(final_data_test$state_changed_at,final_data_test $deadline , units = c("secs"))
deadline_status = as.data.frame(deadline_status)

lunch_created = difftime(final_data_test$launched_at ,final_data_test$created_at  , units = c("secs"))
lunch_created = as.data.frame(lunch_created)

status_created = difftime(final_data_test $state_changed_at ,final_data_test$created_at , units = c("secs"))
status_created  = as.data.frame(status_created)

status_launched = difftime(final_data_test $state_changed_at ,final_data_test$launched_at , units = c("secs"))
status_launched   = as.data.frame(status_launched )


deadline_created = difftime(final_data_test $deadline, final_data_test $created_at , units = c("secs"))
deadline_created = as.data.frame(deadline_created)


deadline_launched_at = difftime(final_data_test $deadline, final_data_test $launched_at , units = c("secs"))
deadline_launched_at = as.data.frame(deadline_launched_at)



final_data_test = cbind (deadline_status,lunch_created,status_created, status_launched ,
                         deadline_created,deadline_launched_at
                         ,final_data_test[,c("goal","disable_communication","country","currency")])

remove(deadline_status,lunch_created,status_created, status_launched ,
       deadline_created,deadline_launched_at)


final_data_test$deadline_status = as.numeric(final_data_test$deadline_status)
final_data_test$lunch_created = as.numeric(final_data_test$lunch_created)
final_data_test$status_created = as.numeric(final_data_test$status_created)
final_data_test$status_launched = as.numeric(final_data_test$status_launched)
final_data_test$deadline_created = as.numeric(final_data_test$deadline_created)
final_data_test$deadline_launched_at = as.numeric(final_data_test$deadline_launched_at)
final_data_test$disable_communication = as.factor(final_data_test$disable_communication)
final_data_test$country = as.factor(final_data_test$country)
final_data_test$currency = as.factor(final_data_test$currency)


final_data_test =  model.matrix(~.+0,data = final_data_test)

final_data_test = as.data.frame(final_data_test)
final_data_test=as.matrix(final_data_test)

dtest <- xgb.DMatrix(data = final_data_test) 



pred <-predict(xgb1, dtest)

pred = as.data.frame(pred)

#combining results with its id which we have extracted earlier
output = cbind(project_id,pred)

#renaming the tareget variable as mentioned in train dataset
names(output)[2] = "final_status"

output$final_status = ifelse(output$final_status > .50,1,0)
output$final_status = as.integer(output$final_status)

#exporting the prediction file as csv
write.csv(output,"xgb.csv",row.names = F)
                
                
                
                #################################### Kaam 25 #############################
                rm(list = ls())
setwd("C:/Users/Hemant.Sain/Desktop/zs/dataset")

library(data.table)
holi = fread("holidays.csv", header = T , stringsAsFactors = FALSE, check.names = FALSE,na.strings=c("","NA"))
ex= fread("promotional_expense.csv", header = T , stringsAsFactors = FALSE, check.names = FALSE,na.strings=c("","NA"))
train = fread("yds_train2018.csv", header = T , stringsAsFactors = FALSE, check.names = FALSE,na.strings=c("","NA"))
names(ex)[4]= "Product_ID"

temp = merge(x = train, y = ex, by = c('Year','Month','Country','Product_ID'), all.x = TRUE)

library(splitstackshape)

date = holi$Date

holi = cSplit(holi, "Date", ",")

holi$Date_4 = as.integer(holi$Date_3/7+1)

holi$is_holi = 1

names(holi)[3] = 'Year'
names(holi)[4] = 'Month'
names(holi)[6] = 'Week'


temp2 = merge(x = temp, y = holi, by = c('Year','Month','Week','Country'), all.x = TRUE)



