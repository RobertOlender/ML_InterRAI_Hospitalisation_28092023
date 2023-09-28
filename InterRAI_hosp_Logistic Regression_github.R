#Import libraries and data set.
library(readr)
library(forcats)
library(haven)
library(smotefamily)
library(ROSE)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(caret)
library(pROC)

#Run a model using all features on the train dataset.
crossValSettings <- trainControl(method = "repeatedcv", number = 100, savePredictions = TRUE)
crossVal <- train(as.factor(hospitalisation) ~., 
                  data = train, 
                  family = "binomial", 
                  method ="glm", 
                  trControl = crossValSettings)
pred_train <- predict(crossVal, newdata = train)

#Run a model using all features on the test dataset.
crossValSettings <- trainControl(method = "repeatedcv", number = 100, savePredictions = TRUE)
crossVal <- train(as.factor(hospitalisation) ~., 
                  data = test, 
                  family = "binomial", 
                  method ="glm", 
                  trControl = crossValSettings)
pred_validation <- predict(crossVal, newdata = test)

#Generate a confusion matrix for the test dataset.
confusionMatrix(data = pred_validation, test$hospitalisation, positive = "Y")

#Generate a variable importance plot.
varImp(crossVal)
plot(varImp(crossVal), top = 20)

#Generate an AUC-ROC.
prediction_for_ROC <- predict(crossVal, test, type = "prob")
ROC_lr <- roc(response = test$hospitalisation, predictor = prediction_for_ROC[,2])
ROC_lr
ROC_lr_AUC <- auc(ROC_lr)
ROC_lr_AUC_95ci <- ci.auc(ROC_lr)
print(ROC_lr_AUC_95ci)

plot(ROC_lr, 
     main = "ROC Curve for the Logistic Regression model", 
     col = "cadetblue4",
     lty = 1,
     lwd = 2,
     asp = NA,
     axes = TRUE,
     print.auc = TRUE)
