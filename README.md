# My-Practical-Machine-Learning-Project
Practical Machine Learning project Assignment
Chiagozie Umeano
3/29/2021
R Markdown
##INTRODUCTION

This is a project concerned with prediction of the manner in which exercise was done among participants using devices like Jawbone up, Nike FuelBand, and Fitbit. Thes participants were a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.These type of devices are part of the quantified self movement – a group of One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. This project uses data fom accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

##Loading the required libraries

library(caret)
## Loading required package: lattice
## Loading required package: ggplot2
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
## Loading required package: tibble
## Loading required package: bitops
## Rattle: A free graphical interface for data science with R.
## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
library(randomForest)
## randomForest 4.6-14
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## The following object is masked from 'package:rattle':
## 
##     importance
## The following object is masked from 'package:ggplot2':
## 
##     margin
library(gbm)
## Loaded gbm 2.1.8
##Loading the datasets

train_url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

init_org_training_data<-read.csv(url(train_url))
init_org_testing_data<-read.csv(url(test_url))

dim(init_org_training_data)
## [1] 19622   160
dim(init_org_testing_data)
## [1]  20 160
##Cleaning Data 1.

non_zero_var <- nearZeroVar(init_org_training_data)

org_training_data <-init_org_training_data[,-non_zero_var]
org_testing_data <- init_org_testing_data[,-non_zero_var]

dim(org_training_data)
## [1] 19622   100
dim(org_testing_data)
## [1]  20 100
na_val_col <- sapply(org_training_data, function(x) mean(is.na(x))) > 0.95

org_training_data <- org_training_data[,na_val_col == FALSE]
org_testing_data <- org_testing_data[,na_val_col == FALSE]

dim(org_training_data)
## [1] 19622    59
dim(org_testing_data)
## [1] 20 59
org_training_data <- org_training_data[,8:59]
org_testing_data <- org_testing_data[,8:59]

dim(org_training_data)
## [1] 19622    52
dim(org_testing_data)
## [1] 20 52
colnames(org_training_data)
##  [1] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [4] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
##  [7] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [10] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [13] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [16] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [19] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [22] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [25] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [28] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [31] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [34] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [37] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [40] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [43] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [46] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [49] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
## [52] "classe"
colnames(org_testing_data)
##  [1] "pitch_belt"           "yaw_belt"             "total_accel_belt"    
##  [4] "gyros_belt_x"         "gyros_belt_y"         "gyros_belt_z"        
##  [7] "accel_belt_x"         "accel_belt_y"         "accel_belt_z"        
## [10] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"       
## [13] "roll_arm"             "pitch_arm"            "yaw_arm"             
## [16] "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"         
## [19] "gyros_arm_z"          "accel_arm_x"          "accel_arm_y"         
## [22] "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"        
## [25] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"      
## [28] "yaw_dumbbell"         "total_accel_dumbbell" "gyros_dumbbell_x"    
## [31] "gyros_dumbbell_y"     "gyros_dumbbell_z"     "accel_dumbbell_x"    
## [34] "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"   
## [37] "magnet_dumbbell_y"    "magnet_dumbbell_z"    "roll_forearm"        
## [40] "pitch_forearm"        "yaw_forearm"          "total_accel_forearm" 
## [43] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"     
## [46] "accel_forearm_x"      "accel_forearm_y"      "accel_forearm_z"     
## [49] "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"    
## [52] "problem_id"
##Partitioning The Data

inTrain <- createDataPartition(org_training_data$classe, p=0.6, list=FALSE)
training <- org_training_data[inTrain,]
testing <- org_training_data[-inTrain,]

dim(training)
## [1] 11776    52
dim(testing)
## [1] 7846   52
#Decision Tree Model

DT_modfit <- train(classe ~ ., data = training,method="rpart")
DT_prediction <- predict(DT_modfit, testing)
DT_pred_conf <- confusionMatrix(DT_prediction,as.factor(testing$classe))

DT_pred_conf
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1814  396   53  132   70
##          B   53  679   49   93  326
##          C  314  303 1104  678  448
##          D   48  139  150  383    5
##          E    3    1   12    0  593
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5828          
##                  95% CI : (0.5718, 0.5938)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4709          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8127  0.44730   0.8070  0.29782  0.41123
## Specificity            0.8840  0.91767   0.7309  0.94787  0.99750
## Pos Pred Value         0.7359  0.56583   0.3878  0.52828  0.97373
## Neg Pred Value         0.9223  0.87376   0.9472  0.87319  0.88269
## Prevalence             0.2845  0.19347   0.1744  0.16391  0.18379
## Detection Rate         0.2312  0.08654   0.1407  0.04881  0.07558
## Detection Prevalence   0.3142  0.15294   0.3629  0.09240  0.07762
## Balanced Accuracy      0.8484  0.68248   0.7690  0.62284  0.70437
#The plot

rpart.plot(DT_modfit$finalModel, roundint=FALSE)


##Random Forest Model

RF_modfit <- train(classe ~ ., data = training, method = "rf", ntree = 100)
#Random forest model prediction

RF_prediction <- predict(RF_modfit, testing)
RF_pred_conf<-  confusionMatrix(RF_prediction,as.factor(testing$classe))

RF_pred_conf
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227   15    0    0    0
##          B    4 1497    8    0    0
##          C    0    4 1355   20    1
##          D    0    1    5 1264    1
##          E    1    1    0    2 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.992           
##                  95% CI : (0.9897, 0.9938)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9898          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9862   0.9905   0.9829   0.9986
## Specificity            0.9973   0.9981   0.9961   0.9989   0.9994
## Pos Pred Value         0.9933   0.9920   0.9819   0.9945   0.9972
## Neg Pred Value         0.9991   0.9967   0.9980   0.9967   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1908   0.1727   0.1611   0.1835
## Detection Prevalence   0.2858   0.1923   0.1759   0.1620   0.1840
## Balanced Accuracy      0.9975   0.9921   0.9933   0.9909   0.9990
#The plot

plot(RF_pred_conf$table, col = RF_pred_conf$byClass, 
     main = paste("Random Forest - Accuracy Level =",
                  round(RF_pred_conf$overall['Accuracy'], 4)))


#Gradient Boosting Model

GBM_modfit <- train(classe ~ ., data = training, method = "gbm", verbose = FALSE)
GBM_modfit$finalModel
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 51 predictors of which 51 had non-zero influence.
GBM_prediction <- predict(GBM_modfit, testing)

GBM_pred_conf <- confusionMatrix(GBM_prediction,as.factor(testing$classe))
GBM_pred_conf
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2192   55    0    0    0
##          B   29 1422   46    5   18
##          C    6   39 1301   42   15
##          D    4    1   20 1219   21
##          E    1    1    1   20 1388
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9587         
##                  95% CI : (0.9541, 0.963)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9478         
##                                          
##  Mcnemar's Test P-Value : 1.036e-08      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9821   0.9368   0.9510   0.9479   0.9626
## Specificity            0.9902   0.9845   0.9843   0.9930   0.9964
## Pos Pred Value         0.9755   0.9355   0.9273   0.9636   0.9837
## Neg Pred Value         0.9929   0.9848   0.9896   0.9898   0.9916
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2794   0.1812   0.1658   0.1554   0.1769
## Detection Prevalence   0.2864   0.1937   0.1788   0.1612   0.1798
## Balanced Accuracy      0.9861   0.9606   0.9676   0.9704   0.9795
#The plot

plot(GBM_pred_conf$table, col = GBM_pred_conf$byClass, 
     main = paste("Gradient Boosting - Accuracy Level =",
                  round(GBM_pred_conf$overall['Accuracy'], 4)))


#Comparing Random Forest and Gradient Boosting Model

RF_pred_conf$overall
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9919704      0.9898419      0.9897382      0.9938245      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
GBM_pred_conf$overall
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.587051e-01   9.477539e-01   9.540651e-01   9.630003e-01   2.844762e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   1.036084e-08
#Conclusion The Random Forest proves more accurate than Gradient Boosting Model. So Random Forest is selected for final prediction from org_testing_data.

#Final Prediction

Final_RF_prediction <- predict(RF_modfit, org_testing_data )
Final_RF_prediction
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
