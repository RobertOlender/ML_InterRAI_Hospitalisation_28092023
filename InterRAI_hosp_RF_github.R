#Import libraries and data set.
library(readr)
library(dplyr)
library(tidyverse)
library(forcats)
library(tidymodels)
library(haven)
library(smotefamily)
library(ROSE)
library(randomForest)
library(caret)
library(e1071)
library(rpart)
library(pROC)

#Set the computational nuances of the 'train' function.
control <- trainControl(method = "repeatedcv",
                        number = 100,
                        repeats = 2,
                        allowParallel = TRUE)
metric = "Accuracy"
tuneGrid = expand.grid(.mtry = (6)) 

#Create a Random Forest model using the train dataset.
rf_model <- train(hospitalisation~., 
                  data = train, 
                  method = "rf", 
                  metric = metric, 
                  tuneGrid = tuneGrid, 
                  trControl = control)

#Generate a variable importance plot.
varImp(rf_model)
plot(varImp(rf_model), top = 20)

#Predict 30-day hospitalisation using the test dataset.
pred_rf <- predict(rf_model, test) 

#Confusion matrix for the test dataset.
confusionMatrix(data = pred_rf, reference = test$hospitalisation,positive='Y')

#Generate an AUC-ROC.
prediction_for_ROC <- predict(rf_model, test, type = "prob")
ROC_rf <- roc(response = test$hospitalisation, predictor = prediction_for_ROC[,2])
ROC_rf
ROC_rf_AUC <- auc(ROC_rf)
ROC_rf_AUC_95ci <- ci.auc(ROC_rf)
print(ROC_rf_AUC_95ci)

plot(ROC_rf, 
     main = "ROC Curve for the Random Forest model", 
     col = "cadetblue4",
     lty = 1,
     lwd = 2,
     asp = NA,
     axes = TRUE,
     print.auc = TRUE)
