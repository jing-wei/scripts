#####
# Based on Max Kuhn's applied predictive modeling
##########
# data already in caret package
library(caret)
data(segmentationData)
# Check structure
str(segmentationData)
# Get rid of cell identifiers
segmentationData$Cell <- NULL
training <- subset(segmentationData, Case == "Train")
testing <- subset(segmentationData, Case == "Test")
# Remove the Case info
training$Case <- NULL
testing$Case <- NULL
str(training)
# 1009
# Remove Class column
trainX <- training[, names(training) != "Class"]
str(trainX)
# Check what's in preProcess
?preProcess
preProcValues <- preProcess(trainX, method=c("center", "scale"))
preProcValues
# Apply to the dataset
scaledTrain <- predict(preProcValues, trainX)
str(scaledTrain)

# Tree
library(rpart)
# 4 step tree

rpart1 <- rpart(Class ~ ., data = training, 
                control = rpart.control(maxdepth=4))
rpart1
# Use partykit for plotting
library(partykit)
rpart1a <- as.party(rpart1)
plot(rpart1a)

# By default, rpart will conduct as many splits as possible, then use 10–fold
# cross–validation to prune the tree
# Specifically, the “one SE” rule is used: estimate the standard error of
# performance for each tree size then choose the simplest tree within one
# standard error of the absolute best tree size
rpartFull <- rpart(Class ~ ., data=training)
rpartFulla <- as.party(rpartFull)
plot(rpartFulla)

# Test set results
rpartPred <- predict(rpartFull, testing, type="class")
# Check confusion matrix for observed and predicted
confusionMatrix(rpartPred, testing$Class)

# CART: 10 fold CV
# the default CART algorithm uses overall accuracy and the one
# standard–error rule to prune the tree

# Mannually tune the model
# train function in caret package
# tuneLength
# 3 repeats of 10-fold CV
?train
?trainControl
cvCtrl <- trainControl(method="repeatedcv", repeats=3)
train(Class ~ ., data=training, method = "rpart", 
      tuneLength = 30, trControl = cvCtrl)
# ROC curve for pruning
cvCtrl <- trainControl(method="repeatedcv", repeats=3, 
                       summaryFunction = twoClassSummary, 
                       classProbs = TRUE)
set.seed(1)
rpartTune <- train(Class ~ ., data=training, method = "rpart", 
                   tuneLength = 30, metric = "ROC", 
                   trControl = cvCtrl)
rpartTune
# plot on log scale
plot(rpartTune, scales = list(x=list(log = 10)))


# Predict testing
rpartPred2 <- predict(rpartTune, testing)
confusionMatrix(rpartPred2, testing$Class)

# Create ROC curve
library(pROC)
# probabilities
rpartProbs <- predict(rpartTune, testing, type = "prob")
str(rpartProbs)
rpartROC <- roc(testing$Class, rpartProbs[, "PS"], 
                levels = rev(testing$Class))
rpartROC
plot(rpartROC, type = "S")
plot(rpartROC, type = "S", print.thres = 0.5)

# C50
library(C50)
grid <- expand.grid(.model = "tree", 
                    .trials = c(1:100), 
                    .winnow = FALSE)
c5Tune <- train(trainX, training$Class, 
                method = "C5.0", metric = "ROC", 
                tuneGrid = grid, trControl = cvCtrl)
c5Tune
plot(c5Tune)
# Predict 
c5Pred <- predict(c5Tune, testing)
confusionMatrix(c5Pred, testing$Class)
# Probs
c5Probs <- predict(c5Tune, testing, type="prob")
str(c5Probs)
# ROC
library(pROC)
c5ROC <- roc(predictor= c5Probs$PS, 
             response = testing$Class, 
             levels = rev(levels(testing$Class)))
c5ROC
# plot earlier curve
plot(rpartROC, type = "S")
# add on new one
plot(c5ROC, add = TRUE, col = "#9E0142")

histogram(~c5Probs$PS|testing$Class, xlab = "Probability of Poor Segmentation")

