#Importing all the required libraries
#install.packages("gridExtra")
# #install.packages("corrplot")
# install.packages("CatEncoders")
# install.packages("plyr")
# install.packages("e1071")
# install.packages("randomForest")
library(ggplot2)
library(gridExtra)
library(corrplot)
library(CatEncoders)
library(plyr)
library(dplyr)
library(class)
library(e1071)
library(rpart)
library(randomForest)

#Data Pre processing
train_dataset = read.csv('Train_data.csv')
test_dataset = read.csv('Test_data.csv')

#Check the structure of the data
summary(train_dataset)
str(train_dataset)
head(train_dataset, n=5)

#Drop phone number from the dataset as the feature provides no significance towards the prediction
train_dataset$phone.number = NULL
test_dataset$phone.number = NULL

#Visualising churn by categorical variables
plot1 = ggplot(train_dataset, aes(train_dataset$international.plan, fill = train_dataset$Churn))+ 
  geom_bar(position = "fill")+labs(x = "International Plan", y="")+
  theme(legend.position = "none")+
  ggtitle("CHURN(Green) & Non-Churn(Red)")

plot2 = ggplot(train_dataset, aes(train_dataset$voice.mail.plan, fill = train_dataset$Churn))+ 
  geom_bar(position = "fill")+labs(x = "Voice Mail Plan", y="")+
  theme(legend.position = "none")+
  ggtitle("CHURN(Green) & Non-Churn(Red)")
#Arragne the plots in a grid for better presentation
grid.arrange(plot1, plot2, nrow = 1)

#Checking the correlation plot to check for highly correlated features
corrplot(cor(train_dataset[sapply(train_dataset, is.numeric)]))

#Removing highly correlated features
train_dataset$total.day.calls=NULL
train_dataset$total.eve.calls=NULL
train_dataset$total.night.calls=NULL
train_dataset$total.intl.calls=NULL
test_dataset$total.day.calls=NULL
test_dataset$total.eve.calls=NULL
test_dataset$total.night.calls=NULL
test_dataset$total.intl.calls=NULL
#Removing state feature
train_dataset$state=NULL
test_dataset$state=NULL

#Encoding all the categorical variables
train_dataset$Churn = factor(train_dataset$Churn, 
                             levels = c(" False.", " True."),
                             labels = c(0,1))
train_dataset$Churn = factor(train_dataset$Churn, levels = c(0,1))

factor = names(which(sapply(train_dataset[-15], is.factor)))
for(i in factor){
  encode = LabelEncoder.fit(train_dataset[,i])
  train_dataset[,i] = transform(encode, train_dataset[,i])
}
testFactor = names(which(sapply(test_dataset[-15], is.factor)))
for(i in testFactor){
  encode = LabelEncoder.fit(test_dataset[,i])
  test_dataset[,i] = transform(encode, test_dataset[,i])
}

#Scaling the training set and the test set
train_dataset[, 1:15] = scale(train_dataset[,1:15])
test_dataset[, 1:15] = scale(test_dataset[, 1:15])
train_dataset = as.data.frame(train_dataset)
test_dataset = as.data.frame(test_dataset)
#The data is processed now lets fit the model in the data

#Function to compute accuracy of model
accuracy = function(cm){
  accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  return(accuracy)
}

#Logistic Regression
logisticClassifier = glm(formula = Churn ~ .,
                         family = binomial,
                         data = train_dataset)

prob_Pred = predict(logisticClassifier, type = 'response', newdata = test_dataset)
logisticPred = ifelse(prob_Pred> 0.5, 1, 0)

#Making the confusion matrix
logisticConfusionMatrix = table(test_dataset$Churn, logisticPred)

#Print the accuracy of logistic regression
accuracyL = accuracy(logisticConfusionMatrix)*100
cat("Accuracy of logistic regression is: " , accuracyL)

#KNN Classifier
kNNPred = knn(train = train_dataset[, -15],
              test = test_dataset[, -15],
              cl = train_dataset[,15],
              k=5)

#Confusion Matrix for KNN
kNNConfusionMatrix = table(test_dataset$Churn, kNNPred)

#Print the accuracy for KNN
accuracyKNN = accuracy(kNNConfusionMatrix)*100
cat("Accuracy of KNN is: " , accuracyKNN)

#SVM Classifier
svmClassifier = svm(formula = Churn ~ .,
                    data = train_dataset,
                    type = "C-classification",
                    kernel  = "radial")

svmPred = predict(svmClassifier, newdata = test_dataset)

#Making confusion matrix for SVM
svmConfusionMatrix = table(test_dataset$Churn, svmPred)

#Print the accuracy of the SVM model
accuracySVM = accuracy(svmConfusionMatrix)*100
cat("Accuracy of SVM is: " , accuracySVM)

#Naive Bayes Classifier
naiveBayesClassifier = naiveBayes(x = train_dataset[-15],
                                  y = train_dataset$Churn)

naiveBayesPred = predict(naiveBayesClassifier, newdata = test_dataset[-15])

#Making the confusion Matrix of the Naive Bayes Classifier
naiveBayesConfusioinMatrix = table(test_dataset$Churn, naiveBayesPred)

#Print the accuracy of the Naive Bayes
accuracyNB = accuracy(naiveBayesConfusioinMatrix)*100
cat("Accuracy of Naive Bayes is: " , accuracyNB)

#Decision Tree Classifier
decisionTreeClassifier = rpart(formula = Churn ~ ., 
                               data = train_dataset)

decisionTreePred = predict(decisionTreeClassifier, newdata = test_dataset[-15], type = 'class')

#Confusion matrix of decison tree
decsionTreeConfusionMatrix = table(test_dataset$Churn, decisionTreePred)

#Print the accuracy of the Decision Tree model
accuracyDecsionTree = accuracy(decsionTreeConfusionMatrix)*100
cat("Accuracy of Decision Tree is: " , accuracyDecsionTree)

#Random Forest Classifier
randomForestClassifier = randomForest(x = train_dataset[-15],
                                      y = train_dataset$Churn,
                                      ntree = 30)

randomForestPred = predict(randomForestClassifier, newdata = test_dataset[-15])

#Confusion Matrix for Random Forest
randomForestConfusionMatrix = table(test_dataset$Churn, randomForestPred)

#Print the accuracy of the Random Forest
accuracyRandomForest = accuracy(randomForestConfusionMatrix)*100
cat("Accuracy of Decision Tree is: " , accuracyRandomForest)
