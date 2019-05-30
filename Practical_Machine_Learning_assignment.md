---
title: 'Practical Machine Learning : Prediction Assignment Writeup'
author: "Johnson Kamireddy"
date: "May 29, 2019"
output: 
    html_document:
        keep_md: TRUE
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
## [1] 13737   102
```

```r
dim(testdata)
```

```
## [1] 5885  102
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
## A 3904    1    0    0    1 0.0005120328
## B    3 2652    2    1    0 0.0022573363
## C    0    6 2390    0    0 0.0025041736
## D    0    0   15 2237    0 0.0066607460
## E    0    0    0    7 2518 0.0027722772
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
##          A 1674    3    0    0    0
##          B    0 1135    1    0    2
##          C    0    1 1025    4    0
##          D    0    0    0  960    7
##          E    0    0    0    0 1073
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9969          
##                  95% CI : (0.9952, 0.9982)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9961          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9965   0.9990   0.9959   0.9917
## Specificity            0.9993   0.9994   0.9990   0.9986   1.0000
## Pos Pred Value         0.9982   0.9974   0.9951   0.9928   1.0000
## Neg Pred Value         1.0000   0.9992   0.9998   0.9992   0.9981
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1929   0.1742   0.1631   0.1823
## Detection Prevalence   0.2850   0.1934   0.1750   0.1643   0.1823
## Balanced Accuracy      0.9996   0.9979   0.9990   0.9972   0.9958
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
##          A 1512  245   44  103   78
##          B   28  625   75   31   91
##          C   24   75  826  147   82
##          D   88  156   51  639  143
##          E   22   38   30   44  688
## 
## Overall Statistics
##                                           
##                Accuracy : 0.729           
##                  95% CI : (0.7174, 0.7403)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6554          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9032   0.5487   0.8051   0.6629   0.6359
## Specificity            0.8884   0.9526   0.9325   0.9110   0.9721
## Pos Pred Value         0.7629   0.7353   0.7158   0.5933   0.8370
## Neg Pred Value         0.9585   0.8979   0.9577   0.9324   0.9222
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2569   0.1062   0.1404   0.1086   0.1169
## Detection Prevalence   0.3368   0.1444   0.1961   0.1830   0.1397
## Balanced Accuracy      0.8958   0.7507   0.8688   0.7869   0.8040
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
##          A 1670    6    0    0    0
##          B    4 1120   10    3    2
##          C    0   12 1010   12    1
##          D    0    1    6  948   16
##          E    0    0    0    1 1063
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9874          
##                  95% CI : (0.9842, 0.9901)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9841          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9833   0.9844   0.9834   0.9824
## Specificity            0.9986   0.9960   0.9949   0.9953   0.9998
## Pos Pred Value         0.9964   0.9833   0.9758   0.9763   0.9991
## Neg Pred Value         0.9990   0.9960   0.9967   0.9967   0.9961
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1903   0.1716   0.1611   0.1806
## Detection Prevalence   0.2848   0.1935   0.1759   0.1650   0.1808
## Balanced Accuracy      0.9981   0.9897   0.9896   0.9894   0.9911
```

Plotting the results of the consufusion matrix to explain it more visually.


```r
plot(confusion_GBM$table, col = confusion_GBM$byClass, main = paste("GBM Accuracy =", round(confusion_GBM$overall['Accuracy'], 4)))
```

![](Practical_Machine_Learning_assignment_files/figure-html/unnamed-chunk-18-1.png)<!-- -->

# Employing the efficient model to the test data:

The accuracy ratings of the 3 models used are as follows:
I.   Random Forest - 0.9971
II.  Decision Tree - 0.7334
III. Generalized Boosted Model - 0.9854

As the accuracy of Random forest model is more efficient and accurate we will be employing this on the test data set.


```r
predict_test <- predict(RFmodfit, newdata = test_data)
predict_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
