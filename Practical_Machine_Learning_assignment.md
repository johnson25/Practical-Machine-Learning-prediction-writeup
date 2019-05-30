---
title: 'Practical Machine Learning : Prediction Assignment Writeup'
author: "Johnson Kamireddy"
date: "May 29, 2019"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---

# OVERVIEW:

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

This report describe how the model was built, how cross validation was used, what expected out of sample error was, and why/ how the specific choices were made. We used prediction model to predict 20 different test cases.

# Loading and Cleaning data:

Loading the required R libraries to perform the analysis.


```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Registered S3 methods overwritten by 'ggplot2':
##   method         from 
##   [.quosures     rlang
##   c.quosures     rlang
##   print.quosures rlang
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## corrplot 0.84 loaded
```

The datasets are downloaded into the system so the next step involves loading the datasets into Rstudio. The dataset was partioned into two. 70% of the data is considered to be training set to train the algorithms and remaining 30% as test set to test the trained models.


```r
training_data <- read.csv(file = "C:/Users/johns/Documents/pml-training.csv", header = TRUE, sep = ",")
test_data <- read.csv(file = "C:/Users/johns/Documents/pml-testing.csv", header = TRUE, sep = ",")

# Creating the data partitions:

train_partition <- createDataPartition(training_data$classe, p = 0.7, list = FALSE)
traindata <- training_data[train_partition, ]
testdata <- training_data[-train_partition, ]

# Dimensions of the training datatset:
dim(traindata)
```

```
## [1] 13737   160
```

```r
# Dimensions of the test datatset:
dim(testdata)
```

```
## [1] 5885  160
```

Both the datasets have 160 variables and there is a possibility that there are plenty NA values present in the dataset.In addition to that the near zero variance variables (NZV) are also removed and the identification variables as well.


```r
zerovariables <- nearZeroVar(traindata)
traindata <- traindata[, -zerovariables]
testdata <- testdata[, -zerovariables]

dim(traindata)
```

```
## [1] 13737   105
```

```r
dim(testdata)
```

```
## [1] 5885  105
```

Removing the variables that have most NA values

```r
navariables <- sapply(traindata, function(x) mean(is.na(x))) > 0.95
traindata <- traindata[, navariables == FALSE]
testdata <- testdata[, navariables == FALSE]

dim(traindata)
```

```
## [1] 13737    59
```


```r
dim(testdata)
```

```
## [1] 5885   59
```

Removing the identification variables which are columns 1 to 5


```r
traindata <- traindata[, -(1:5)]
testdata <- testdata[, -(1:5)]

dim(traindata)
```

```
## [1] 13737    54
```


```r
dim(testdata)
```

```
## [1] 5885   54
```
After the data wrangling we have 54 variables available with 13737 and 5885 records in traindata and testdata datasets.

# Exploratory Data Analysis:

## Correlation Analysis

The correlation analysis is performed on the train dataset before modeling it with machine learning techniques.


```r
cor_matrix <- cor(traindata[, -54])

corrplot(cor_matrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![](Practical_Machine_Learning_assignment_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

# Prediction Modeling:

Three methods are applied to the train dataset to model the regressions and the accurate model with the higher accuracy when applied to test data will be employed for further results or predictions. The methods employed are Random forests, Decision tree and generalized boosted model.

## Random Forests:


```r
set.seed(12345)
RFcontrol <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
RFmodfit <- train(classe ~ ., data = traindata, method="rf", trControl=RFcontrol)

RFmodfit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.26%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    0    0    0    1 0.0002560164
## B    5 2650    3    0    0 0.0030097818
## C    0    9 2385    2    0 0.0045909850
## D    0    0    8 2243    1 0.0039964476
## E    0    2    1    4 2518 0.0027722772
```

Using the trained model on the test dataset to check the accuracy of the model.


```r
predict_RF <- predict(RFmodfit, newdata = testdata)
confusion_RF <- confusionMatrix(predict_RF, testdata$classe)
confusion_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    0
##          B    1 1137    4    0    0
##          C    0    2 1022    7    0
##          D    0    0    0  957    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9956, 0.9984)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9982   0.9961   0.9927   0.9982
## Specificity            1.0000   0.9989   0.9981   0.9996   1.0000
## Pos Pred Value         1.0000   0.9956   0.9913   0.9979   1.0000
## Neg Pred Value         0.9998   0.9996   0.9992   0.9986   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1932   0.1737   0.1626   0.1835
## Detection Prevalence   0.2843   0.1941   0.1752   0.1630   0.1835
## Balanced Accuracy      0.9997   0.9986   0.9971   0.9962   0.9991
```

Plotting the results of the consufusion matrix to explain it more visually.


```r
plot(confusion_RF$table, col = confusion_RF$byclass, main = paste("Random Forest Accuracy = ", round(confusion_RF$overall['Accuracy'], 4)))
```

![](Practical_Machine_Learning_assignment_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

## Decision Trees:


```r
set.seed(12345)
DTmodfit <- rpart(classe ~ ., data = traindata, method = "class")
fancyRpartPlot(DTmodfit)
```

![](Practical_Machine_Learning_assignment_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

Using the trained model on the test dataset to check the accuracy of the model.


```r
predict_DT <- predict(DTmodfit, newdata = testdata, type = "class")
confusion_DT <- confusionMatrix(predict_DT, testdata$classe)
confusion_DT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1438  135   18   47   55
##          B   50  680   37   75  101
##          C   23   66  807  146   83
##          D  141  200   96  636  142
##          E   22   58   68   60  701
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7242          
##                  95% CI : (0.7126, 0.7356)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6516          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8590   0.5970   0.7865   0.6598   0.6479
## Specificity            0.9394   0.9446   0.9346   0.8823   0.9567
## Pos Pred Value         0.8494   0.7211   0.7173   0.5235   0.7712
## Neg Pred Value         0.9437   0.9071   0.9540   0.9298   0.9234
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2444   0.1155   0.1371   0.1081   0.1191
## Detection Prevalence   0.2877   0.1602   0.1912   0.2065   0.1545
## Balanced Accuracy      0.8992   0.7708   0.8606   0.7710   0.8023
```

Plotting the results of the consufusion matrix to explain it more visually.


```r
plot(confusion_DT$table, col = confusion_DT$byclass, main = paste("Decision Tree Accuracy = ", round(confusion_DT$overall['Accuracy'], 4)))
```

![](Practical_Machine_Learning_assignment_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

## Generalized Boosted Model:


```r
set.seed(12345)
GBMcontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
GBMmodfit <- train(classe ~ ., data = traindata, method = "gbm", trControl = GBMcontrol, verbose = FALSE)

GBMmodfit$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 53 had non-zero influence.
```


```r
predict_GBM <- predict(GBMmodfit, newdata = testdata)
confusion_GBM <- confusionMatrix(predict_GBM, testdata$classe)
confusion_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    9    0    0    0
##          B    3 1123    9    6    2
##          C    0    7 1011    6    2
##          D    0    0    4  951   13
##          E    0    0    2    1 1065
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9891          
##                  95% CI : (0.9861, 0.9916)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9862          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9860   0.9854   0.9865   0.9843
## Specificity            0.9979   0.9958   0.9969   0.9965   0.9994
## Pos Pred Value         0.9946   0.9825   0.9854   0.9824   0.9972
## Neg Pred Value         0.9993   0.9966   0.9969   0.9974   0.9965
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1908   0.1718   0.1616   0.1810
## Detection Prevalence   0.2855   0.1942   0.1743   0.1645   0.1815
## Balanced Accuracy      0.9980   0.9909   0.9911   0.9915   0.9918
```

Plotting the results of the consufusion matrix to explain it more visually.


```r
plot(confusion_GBM$table, col = confusion_GBM$byClass, main = paste("GBM Accuracy =", round(confusion_GBM$overall['Accuracy'], 4)))
```

![](Practical_Machine_Learning_assignment_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

# Employing the efficient model to the test data:

The accuracy ratings of the 3 models used are as follows:
I.   Random Forest - 0.99
II.  Decision Tree - 0.73
III. Generalized Boosted Model - 0.98

As the accuracy of Random forest model is more efficient and accurate we will be employing this on the test data set.


```r
predict_test <- predict(RFmodfit, newdata = test_data)
predict_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
