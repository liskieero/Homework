###########################################################################
# load packages
###########################################################################

# Check that required packages are installed
need <- c("caret","doParallel","class","gmodels")
have <- need %in% rownames(installed.packages())
if ( any(!have) ) { install.packages( need[!have] ) }

# load packages
library(caret)
library(doParallel)
library(class)
library(gmodels)



###########################################################################
## function
###########################################################################

# fit control
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)


####
## a function that calculates the performance for each predictor set combination
cv.auto <- function(data, x.names, y.names) {
  
  # start time
  start <- Sys.time()
  
  # set empty result vector for accuracy, predictor names and id index
  output.acc <- NULL
  output.x <- list()
  output.index <- NULL
  
  # define data
  DATA <- data
  x.names <- x.names
  y.names <- y.names
  x.length <- length(x.names)
  y.vec <- DATA[,y.names]
  
  ###
  # main cv loop
  for(i in 1:x.length) {
    # take a predictor combination
    comb <- combn(x.names, x.length-(i-1))
    ncol.c <- ncol(comb)
    j <- 1
    
    while(j <= ncol.c) {
      
      # predictors
      x.df <- DATA[,comb[,j]]
      
      # open cluster for parallel computing
      #registerDoParallel(cores = 4)
      
      # train model
      fit.1 <- train(data.frame(x.df), y.vec, 
                     method = "glm", family="binomial", 
                     trControl = fitControl)
      
      # close cluster
      #stopImplicitCluster()
      
      # results
      out.x.len <- length(output.x)
      output.acc[out.x.len + 1] <- fit.1$results$Accuracy
      output.x[[out.x.len + 1]] <- comb[,j]
      output.index[out.x.len + 1] <- out.x.len + 1
      j <- j + 1
      
    }
  }
  
  
  # total time
  end <- Sys.time()
  total <- end - start
  
  # return
  names(output.acc) <- 1:length(output.index)
  list(accuracy = output.acc, x.names = output.x, index = output.index, runtime = total)
  
}




########################################################################################
## run function
########################################################################################

# define predictors and response
x.names <- c("Q29","Q3","Q10_1","Q10_2","Q10_3","Q10_4","sukupuoli","koulutus","ammatti",
             "ikaryhma","itapohjoinen","asuumaalla","luonnons")
y.names <- "Q19mod"

# run function
set.seed(1)
output.cvauto <- cv.auto(data = maa, x.names = x.names, y.names = y.names)

# find top predictor set and the corresponding accuracy
top.ind <- as.integer(names(sort(output.cvauto$accuracy, decreasing=TRUE)[1]))
outp.1 <- output.cvauto$x.names[top.ind]
top.acc.1 <- output.cvauto$accuracy[top.ind]



########################################################################################
## run knn
########################################################################################

# define features and response
x.full <- maa[, x.names]
y.full <- maa[, y.names]

# pre-process predictors
x.mod.form <-  as.formula(paste("", paste(x.names, collapse=" + "), sep=" ~ "))
x.model <- model.matrix(x.mod.form, data = x.full)[, c(-1)]
df.comb <- data.frame(y.full = y.full, x.model = x.model)
# knn imputation
df.comb <- knnImputation(df.comb, k = 10, scale = FALSE)


## knn fit
knnfit <- train(factor(y.full) ~ .,
                data = df.comb,
                method = "knn",
                trControl = fitControl,
                tuneGrid = data.frame(.k = 50:70))

knnfit

