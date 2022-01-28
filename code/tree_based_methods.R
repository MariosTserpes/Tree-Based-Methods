#04 - ASSIGNMENT MACHINE LEARNING ALGORITHMS WITH R#
             ###TSERPES MARIOS###
                  #AM: 20029#

#Necessary Libraries
install.packages("tree")
install.packages("ISLR")
install.packages("randomForest")
library(tree)
library(ISLR)
library(caret)
library(randomForest)

#Load Carseats Dataset
?Carseats
attach(Carseats)
View(Carseats)

#EXERCISE 1
#(a) Split the data set into a training set and a test set
set.seed(1)
train_set <- sample(1 : nrow(Carseats), nrow(Carseats) / 2) #Selected randomly 50% of data
carseats.train <- Carseats[train_set, ]
carseats.test <- Carseats[-train_set, ]


#(b) Fit a regression tree to the training set. Plot the tree and interpret the results. What train and test MSE do you obtain? Is there indication of overfitting?
reg.tree <- tree(Sales ~ ., data = carseats.train)
summary(reg.tree)
plot(reg.tree)
text(reg.tree, pretty = 0)
#MSE in test set
yhat <- predict(reg.tree, newdata = carseats.test)
mean((yhat - carseats.test$Sales)^2)
#MSE in train set
mean((predict(reg.tree, newdata = carseats.train) - carseats.train$Sales)^2)
#MSE in train and test
data.frame(MSE.train.set = mean((predict(reg.tree, newdata = carseats.train) - carseats.train$Sales)^2),
           MSE.test.set  = mean((yhat - carseats.test$Sales)^2))

#R2 in test and train set 
data.frame(R2.test.set  = R2(yhat, carseats.test$Sales),
           R2.train.set = R2(predict(reg.tree, newdata = carseats.train), carseats.train$Sales))
#RMSE in test and train set
data.frame(RMSE.test.set  = RMSE(yhat, carseats.test$Sales),
           RMSE.train.set = RMSE(predict(reg.tree, newdata = carseats.train), carseats.train$Sales))


#(c)Use cross-validation in order to determine the optimal level of tree complexity. Does pruning the tree improve the test MSE? What about train MSE?
set.seed(20)
cv.reg.tree <- cv.tree(reg.tree, FUN = prune.tree)
summary(cv.reg.tree)
names(cv.reg.tree)
cv.reg.tree
par(mfrow=c(1, 2))
plot(cv.reg.tree$size, cv.reg.tree$dev, type = "b", col = "red")
plot(cv.reg.tree$k, cv.reg.tree$dev, type = "b", col = "red")


#Pruning the tree based on obtaining terminal nodes by using CV
prune.carseats <- prune.tree(reg.tree, best = 11)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
yhat <- predict(prune.carseats, newdata = carseats.test)
mean((yhat - carseats.test$Sales)^2)

#MSE in train and test set
data.frame(MSE.train.set = mean((predict(prune.carseats, newdata = carseats.train) - carseats.train$Sales)^2),
           MSE.test.set  = mean((yhat - carseats.test$Sales)^2))


#(d) Use the bagging approach in order to analyze this data. What train and what test MSE do 
#you obtain? Use the importance() function to determine which variables are most important.
set.seed(1)
bag.carseats <-  randomForest(Sales~ . , data = Carseats, mtry = 10, 
                              subset = train_set, importance = TRUE)

bag.carseats
yhat.bag <- predict(bag.carseats, newdata = Carseats[-train_set, ])
mean((yhat.bag - carseats.test$Sales)^2)

data.frame(MSE.train.set = mean((predict(bag.carseats, newdata = carseats.train) - carseats.train$Sales)^2),
           MSE.test.set  = mean((yhat.bag - carseats.test$Sales)^2))

data.frame(R2.train.set = R2(predict(bag.carseats, newdata = carseats.train), carseats.train$Sales),
           R2.test.set  = R2(yhat.bag, carseats.test$Sales))


importance(bag.carseats)
varImpPlot(bag.carseats)

#(e) Use random forests to analyze this data. What train and what test MSE do you obtain? Use 
#the importance() function to determine which variables are most important. Describe the 
#effect of m, the number of variables considered at each split, on the error rate obtained. 
set.seed(20)
#Parameter tuning of Number of variables at each split
MSE.rf <- NA 
for (a in 1 : 10){
  RandomForest <- randomForest(Sales ~ ., data = carseats.train, mtry = a,
                               ntree = 500, importance = TRUE)
  RandomForest.pred <- predict(RandomForest, carseats.test)
  MSE.rf[a] <- mean((RandomForest.pred - carseats.test$Sales)^2)
  }
min(MSE.rf) #refers in 9 variables on each tree

#Tuning Number of Trees
set.seed(20)
MSE.rf2 <- NA 
for (ntree in c(20, 25, 50, 100, 150, 200)){
  RandomForest2 <- randomForest(Sales ~ ., data = carseats.train, mtry = 9,
                               ntree = ntree, importance = TRUE)
  RandomForest2.pred <- predict(RandomForest2, carseats.test)
  MSE.rf2[ntree] <- mean((RandomForest2.pred - carseats.test$Sales)^2)
}

set.seed(20)
RandomForest3 <- randomForest(Sales ~ ., data = carseats.train, mtry = 9,
                              ntree = 50, importance = TRUE)

RandomForest3
yhat.rf <- predict(RandomForest3, newdata = carseats.test)
#Test Error
mean((yhat.rf - carseats.test$Sales)^2)
#Train Error
mean((predict(RandomForest3, newdata = carseats.train) - carseats.train$Sales)^2)

importance(RandomForest3)





#EXERCISE 2
#(a)Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.
?OJ
View(OJ)
str(OJ)
str(OJ)


set.seed(100)
train = sample(nrow(OJ), 800)
train.set = OJ[train, ]
test.set = OJ[-train, ]

#(b)Fit a tree to the training data, with Purchase as the response and the other variables as 
#predictors. Use the summary() function to produce summary statistics about the tree, and 
#describe the results obtained. What is the training error rate? How many terminal nodes does 
#the tree have?

tree.2 = tree(Purchase ~ ., data = train.set)
summary(tree.2)

#(c)Type in the name of the tree object in order to get a detailed text output. Pick one of the terminal nodes, and interpret the information displayed.
tree.2

#(d)Create a plot of the tree, and interpret the results. 
plot(tree.2)
text(tree.2, pretty = 0)


#(e) Predict the response on the test data, and produce a confusion matrix comparing the test 
#labels to the predicted test labels. What is the test error rate? Repeat the same for the train 
#data set, as well.
#TEST SET
tree.pred <- predict(tree.2, test.set, type = "class")
table(tree.pred, test.set$Purchase)
1 - (142 + 69) / nrow(test.set)
#TRAIN SET
tree.pred2 <- predict(tree.2, train.set, type = "class")
table(tree.pred2, train.set$Purchase)
1 - (432 + 241) / nrow(train.set)

#(f)Apply the cv.tree() function to the training set in order to determine the optimal tree size.
set.seed(100)
cv.oj <- cv.tree(tree.2, FUN = prune.misclass)
names(cv.oj)

#(g)Produce a plot with tree size on the $x$-axis and cross-validated classification error rate on the $y$-axis
plot(cv.oj$size, cv.oj$dev, type = "b", xlab = "size", ylab = "Deviance",
     col = "red")


#(i)Produce a pruned tree corresponding to the optimal tree size 
#obtained using cross-validation. If cross-validation does not 
#lead to selection of a pruned tree, then create a pruned 
#tree with five terminal nodes
prune.oj <- prune.misclass(tree.2, best = 6)
plot(prune.oj)
text(prune.oj, pretty = 0)

#(j)Compare the training error rates between the pruned and unpruned trees. Which is higher?Explain! 
summary(tree.2)
tree.2
summary(prune.oj)
prune.oj


#(k) Compare the test error rates between the pruned and unpruned trees. Which is higher? Explain!
pred.unpruned <-  predict(tree.2, test.set, type="class")
misclass.unpruned <-  sum(test.set$Purchase != pred.unpruned)
misclass.unpruned / length(pred.unpruned)
