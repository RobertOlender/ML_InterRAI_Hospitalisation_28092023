#Import libraries and data set.
library(readr)
library(dplyr)
library(tidyverse)
library(forcats)
library(haven)
library(smotefamily)
library(ROSE)
library(ggplot2)
library(xgboost)
library(caret)
library(mltools)
library(pROC)

#Grid tune for hyperparameter optimisation.
grid_tune <- expand.grid(
  nrounds = c(500, 750, 1000), 
  max_depth = c(2,3,4),
  eta = c(0.05, 0.1, 0.3),
  gamma = c(0, 0.1, 0.5, 1.0),
  colsample_bytree = c(0.5, 1.0),
  min_child_weight = c(1, 1.5, 2),
  subsample = c(0.5, 1.0))


train_control <- trainControl(method = "repeatedcv",
                              number = 100,
                              repeats = 2,
                              allowParallel = TRUE)

xgb_tune <- train(x = train[,-85],
                  y = train[,85],
                  trControl = train_control,
                  tuneGrid = grid_tune,
                  method = "xgbTree")

xgb_tune

#Find the best tune.
xgb_tune$bestTune

#Write out the best model.
train_control <- trainControl(method = "none",
                              allowParallel = TRUE)

final_grid <- expand.grid(nrounds = xgb_tune$bestTune$nrounds,
                          eta = xgb_tune$bestTune$eta,
                          max_depth = xgb_tune$bestTune$max_depth,
                          gamma = xgb_tune$bestTune$gamma,
                          colsample_bytree = xgb_tune$bestTune$colsample_bytree,
                          min_child_weight = xgb_tune$bestTune$min_child_weight,
                          subsample = xgb_tune$bestTune$subsample)

xgb_model <- train(x = train[,-85],
                   y = train[,85],
                   trControl = train_control,
                   tuneGrid = final_grid,
                   method = "xgbTree")

#Prediction using the test dataset.
xgb.pred <- predict(xgb_model, test)

#Confusion matrix for the test dataset.
confusionMatrix(as.factor(as.numeric(xgb.pred)),
                as.factor(as.numeric(test$lab_test)),
                positive = "1")

#Generate a variable importance plot.
varImp(xgb_model)
plot(varImp(xgb_model), top = 20)

#Generate an AUC-ROC.
prediction_for_ROC <- predict(xgb_model, test, type = "prob")
ROC_xgb <- roc(response = test$lab_test, predictor = prediction_for_ROC[,2])
ROC_xgb
ROC_xgb_AUC <- auc(ROC_xgb)
ROC_xgb_AUC_95ci <- ci.auc(ROC_xgb)
print(ROC_xgb_AUC_95ci)

plot(ROC_xgb, 
     main = "ROC Curve for the XGBoost model", 
     col = "cadetblue4",
     lty = 1,
     lwd = 2,
     asp = NA,
     axes = TRUE,
     print.auc = TRUE)
