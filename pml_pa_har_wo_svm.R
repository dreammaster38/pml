
# Load all necessary libraries used for this project
library(caret)
library(randomForest)
library(e1071)
library(doParallel)


## Step 1: Loading data
#Load the data sets and replace unuseful strings with NAs.
trainRawData <- read.csv("data/pml-training.csv", na.strings=c("NA","", "#DIV/0!"))
testingRawData <- read.csv("data/pml-testing.csv", na.strings=c("NA","", "#DIV/0!"))

## Step 2: Cleaning up the data

set.seed(1972)
cleanedData <- trainRawData[ ,colSums(is.na(trainRawData)) == 0]
testing <- testingRawData[ ,colSums(is.na(trainRawData)) == 0]

# data partitioning into training and cross validation set
trainIndex <- createDataPartition(y = cleanedData$classe, p=0.2, list=FALSE) # 3927 rows
training <- cleanedData[trainIndex,]
cross <- cleanedData[-trainIndex,]

# Discard unuseful predictors because they are not numeric this gives us 53 predictors
columnsToRemove <- names(training) %in% c("raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "X", "user_name", "new_window")
training <- training[ , !columnsToRemove]
cross <- cross[ , !columnsToRemove]
testing <- testing[, !columnsToRemove]

# check data for skewness
classeName <- names(training) %in% c("classe") 
testForSkewness <- training[!classeName]
# apply the skewnes function to each numeric column of our training set
skewValues <- apply(testForSkewness, 2, skewness)
# create a data frame for fancier printing
skewValuesDf <- data.frame(skewValues)
print(skewValuesDf)
# plot a histogram of the skewness.
hist(skewValues, col=heat.colors(17), xlab="Skewness of all predictors", breaks=20)

# Step 3 Model creation

# Build a Random Forest (RF)

# Create clusters for all available cores communicating over sockets
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# model fitting
ctrl <- trainControl(method='cv', number=10, allowParallel=TRUE)
modFitRf <- train(training$classe ~.,
data = training,
do.trace=100,
method="rf",
trControl=ctrl,
preProcess=(method=c("center", "scale")))


# Results
# show OOB
print(modFitRf)
print(modFitRf$finalModel)

# predict against our cross validation set created in step 2 to find out the accuracy of our model.
predCrossRf <- predict(modFitRf, cross)
print(confusionMatrix(predCrossRf, cross$classe))

# Print the overall agreement and Kappa:

accuracySummary <- postResample(predCrossRf, cross$classe)
print(accuracySummary)

# prediction results on the supplied test set for Random Forest
predRf <- predict(modFitRf, testing)
print(predRf)

# stop all created cluster nodes
stopCluster(cl)