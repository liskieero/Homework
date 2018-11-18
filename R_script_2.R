##################################################################################################
## load packages
##################################################################################################

# Check that required packages are installed
need <- c("caret","MXM","tidyverse","klaR","class","gmodels","C50","kernlab")
have <- need %in% rownames(installed.packages())
if ( any(!have) ) { install.packages( need[!have] ) }

# load packages
library(caret) # automated machine learning
library(MXM) # forward selection
library(tidyverse)
library(klaR) # naive bayes
library(class) # knn
library(gmodels) # Crosstable
library(C50) # 5.0 decision tree
library(kernlab) # support vector machines





##################################################################################################
## read data
##################################################################################################

# define path (change accordingly)
main <- "<path>"
setwd(main)

# read train data
data.train <- read.csv("train_data.csv", header = FALSE)

# read train data labels
data.train.labels <- read.csv("train_labels.csv", header = FALSE)

# read test data
data.test <- read.csv("test_data.csv", header = FALSE)





##################################################################################################
## pre-processing
##################################################################################################

# general examination
str(data.train)
sum(apply(data.train, 2, class) == "numeric")

# examine missing values
no.missing.train <- sum(is.na(data.train)) 
no.missing.test <- sum(is.na(data.test))

# count the number of duplicated rows 
duplicated(data.train) %>%
  sum()
duplicated(data.test) %>%
  sum()

# calculate pairwise correlations
pairw.cor <- data.train %>% 
  cor %>%
  as.data.frame %>%
  rownames_to_column(var = 'var1') %>%
  gather(var2, value, -var1)
# which of the correlations exceed 0.9
which(abs(pairw.cor$value) > 0.9)

# modify response (as factor)
table(data.train.labels)
Y.f <- factor(data.train.labels$V1, levels = c(-1, 1), labels = c("no", "yes"))





##################################################################################################
## dimension reduction
##################################################################################################

# run pca
res.pca <- prcomp(data.train, scale = TRUE)

# pick number of principal components
screeplot(res.pca, npcs = 200) 
n.pc <- min(which(summary(res.pca)$importance[3,] > 0.7))

# rotated data
rot.x <- res.pca$x[, 1:n.pc]
rot.x <- as.data.frame(rot.x)

# forward selection
m.forw.glm <- glm.fsreg(target = Y.f, dataset = data.train, ncores = 1)
x.forw <- names(m.forw.glm$final$data)





##################################################################################################
## train models
##################################################################################################

# no information rate
noinf.rate <- prop.table(table(Y.f))[2] 

# fit control (for train function)
fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 10)


# logit with stepwise AIC predictors
fit.1 <- train(x = data.train[x.forw], y = Y.f, 
               method = "glm", family="binomial", 
               trControl = fitControl)
fit.1

# logit with pca reduced data
fit.2 <- train(x = rot.x, y = Y.f, 
               method = "glm", family="binomial", 
               trControl = fitControl) # convergence problems
fit.2

# naive bayes with pca reduced data
fit.3 <- train(x = rot.x, y = Y.f,
               method = "nb",
               trControl = fitControl)
fit.3

# k nearest neighbors (with scaling)
fit.4 <- train(x = data.train, y = Y.f,
               method = "knn",
               trControl = fitControl,
               preProcess = c("center", "scale"),
               tuneGrid = data.frame(.k = 55:65)) 
fit.4

# k nearest neighbors (without scaling)
fit.5 <-  train(x = data.train, y = Y.f,
                method = "knn",
                trControl = fitControl,
                preProcess = NULL,
                tuneGrid = data.frame(.k = 55:65))
fit.5



#############################################################
## support vector machines (svm) with different tuning

# random seed
set.seed(1)
# create 10 folds
folds <- createFolds(Y.f, k = 10)

# run svm with linear and radial kernels with 10-fold cross validation
cv.res.svm <- lapply(folds, function(x) {
  
  # define train and test sets
  x.train <- data.train[-x, ]
  y.train <- Y.f[-x]
  x.test <- data.train[x, ]
  y.test <- Y.f[x]
  
  # run models with different tuning
  m.svm.lin.C01 <- ksvm(x = as.matrix(x.train), y = y.train, kernel = "vanilladot", C = 0.1)
  m.svm.lin.C1 <- ksvm(x = as.matrix(x.train), y = y.train, kernel = "vanilladot", C = 1)
  m.svm.lin.C10 <- ksvm(x = as.matrix(x.train), y = y.train, kernel = "vanilladot", C = 10)
  m.svm.rad.s005.C1 <- ksvm(x = as.matrix(x.train), y = y.train, kernel = "rbfdot", C = 1, 
                            kpar = list(sigma = 0.05))
  m.svm.rad.s005.C10 <- ksvm(x = as.matrix(x.train), y = y.train, kernel = "rbfdot", C = 10, 
                            kpar = list(sigma = 0.05))
  m.svm.rad.sdef.C1 <- ksvm(x = as.matrix(x.train), y = y.train, kernel = "rbfdot", C = 1)
  m.svm.rad.sdef.C10 <- ksvm(x = as.matrix(x.train), y = y.train, kernel = "rbfdot", C = 10)
  
  # predictions and accuracies
  y.pred.svm.lin1 <- predict(m.svm.lin.C01, x.test)
  y.pred.svm.lin2 <- predict(m.svm.lin.C1, x.test)
  y.pred.svm.lin3 <- predict(m.svm.lin.C10, x.test)
  y.pred.svm.rad1 <- predict(m.svm.rad.s005.C1, x.test)
  y.pred.svm.rad2 <- predict(m.svm.rad.s005.C10, x.test)
  y.pred.svm.rad3 <- predict(m.svm.rad.sdef.C1, x.test)
  y.pred.svm.rad4 <- predict(m.svm.rad.sdef.C10, x.test)
  acc.lin1 <- confusionMatrix(y.pred.svm.lin1, y.test)$overall[1]
  acc.lin2 <- confusionMatrix(y.pred.svm.lin2, y.test)$overall[1]
  acc.lin3 <- confusionMatrix(y.pred.svm.lin3, y.test)$overall[1]
  acc.rad1 <- confusionMatrix(y.pred.svm.rad1, y.test)$overall[1]
  acc.rad2 <- confusionMatrix(y.pred.svm.rad2, y.test)$overall[1]
  acc.rad3 <- confusionMatrix(y.pred.svm.rad3, y.test)$overall[1]
  acc.rad4 <- confusionMatrix(y.pred.svm.rad4, y.test)$overall[1]
  return(list(acc.lin1, acc.lin2, acc.lin3, acc.rad1, acc.rad2, acc.rad3, acc.rad4))
  
})

# mean accuracies
lapply(cv.res.svm, mean)



#############################################################
## svm using train function

# specify different costs
user.grid.lin <- expand.grid(C = c(0,1,10,100))
user.grid.rad <- expand.grid(sigma = c(0.01, 0.1, 0.5),
                             C = c(0,1,10,100))

# run svm with linear kernel
svm.linear <- train(x = data.train, y = Y.f,
                    method = "svmLinear",
                    preProc = c("center", "scale"),
                    metric = "Accuracy",
                    trControl = fitControl,
                    tuneGrid = user.grid.lin)

# run svm with radial kernel
svm.radial <- train(x = data.train, y = Y.f,
                    method = "svmLinear",
                    preProc = c("center", "scale"),
                    metric = "Accuracy",
                    trControl = fitControl,
                    tuneGrid = user.grid.rad)



#############################################################
## c5.0 decision tree with manual cross validation

# random seed
set.seed(1)

# create 10 folds
folds <- createFolds(Y.f, k = 10)

# run c5.0 with boosting and winnow and 10-fold cross validation
cv.res.c5.w <- lapply(folds, function(x) {
  
  # define train and test set
  x.train <- data.train[-x, ]
  y.train <- Y.f[-x]
  x.test <- data.train[x, ]
  y.test <- Y.f[x]
  
  # run model
  m.c5 <- C5.0(x = x.train, y = y.train, trials = 5, control = C5.0Control(winnow = TRUE))
  
  # predict and get accuracy
  y.pred <- predict(m.c5, x.test)
  res.table <- CrossTable(x = y.test, y = y.pred, prop.chisq = FALSE)
  acc <- sum(diag(res.table$prop.tbl))
  return(acc)
  
})

# final accuracy
c5w.acc <- mean(unlist(cv.res.c5.w))
c5w.acc



###
# run c5.0 with boosting and 10-fold cross validation
cv.res.c5 <- lapply(folds, function(x) {
  
  # define train and test set
  x.train <- data.train[-x, ]
  y.train <- Y.f[-x]
  x.test <- data.train[x, ]
  y.test <- Y.f[x]
  
  # run model
  m.c5 <- C5.0(x = x.train, y = y.train, trials = 5, control = C5.0Control(winnow = FALSE))
  
  # predict and get accuracy
  y.pred <- predict(m.c5, x.test)
  res.table <- CrossTable(x = y.test, y = y.pred, prop.chisq = FALSE)
  acc <- sum(diag(res.table$prop.tbl))
  return(acc)
  
})

# final accuracy
c5.acc <- mean(unlist(cv.res.c5))
c5.acc



#############################################################
## C5.0 using train function (memory problems)

# tuning
user.grid <- expand.grid( .winnow = c(TRUE, FALSE), .trials = c(1,5,10), .model = c("tree"))

# run model
set.seed(1)
fit.c5 <- train(x = data.train, y = Y.f,
                method = "C5.0",
                tuneGrid = user.grid,
                trControl = fitControl,
                metric = "Accuracy")






##################################################################################################
## final predictions
##################################################################################################

# train C5.0 (with boosting and winnow) using full training data
m.c5.final <- C5.0(x = data.train, y = Y.f, trials = 5, control = C5.0Control(winnow = TRUE))

# predict 
y.pred.final <- predict(m.c5.final, data.test)
test.labels <- as.numeric(y.pred.final)
test.labels <- ifelse(test.labels == 2, 1, -1)
write.table(test.labels, file = "test_labels.csv", row.names = FALSE, col.names = FALSE)
