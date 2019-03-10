#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 01:56:19 2019

@author: gauravmalik
"""

#First we load all the important libraries that will be used 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#Uploading the file 
df = pd.read_csv("Train_data.csv")
testdf = pd.read_csv("Test_data.csv")


#The file is loaded. Now lets check the dimension of the data
print(df.shape)


#Lets start by checking the head of the data
df.head(5)


#Now we check the tail of the data. This is important as we get the gist of data as what type
#of values does it store 
df.tail(5)


#After looking at the data we now need to know structure of the dataset
df.info()
#The dataset has no NA data in the fields which spares us from removing those rows or replacing them with other values


#Statistics can give us a great insight of the shape of each attribute
df.describe()


#As we don't need phone numbers we can simply drop it from the data
df = df.drop(['phone number'], axis = 1)
testdf = testdf.drop(['phone number'], axis = 1)
list(df)


#Data preprocessing

#Checking for the highly correlated features and removing them so the features are independent of each other
data_trainsdf = df.copy()

#These are all the continuous columns in the dataset
cnames = ['account length','number vmail messages', 'total day minutes','total day calls',  'total day charge',
          'total eve minutes', 'total eve calls', 'total eve charge', 'total night minutes', 'total night calls',
          'total night charge', 'total intl minutes', 'total intl calls', 'total intl charge', 'number customer service calls']

#Seeing the correlation of the features to remove highly correlated features
df_corr = data_trainsdf.loc[:,cnames]
corr = df_corr.corr()

#Plotting a heatmap to see all the correlated features
plt.figure(figsize = (7,7))
plt.title("Heat Map showing Correlation between Features")
sn.heatmap(corr, mask=np.zeros_like(corr, dtype = np.bool),cmap=sn.diverging_palette(220, 10, as_cmap=True), square = True)

#From the heatmap it is evident that the minutes and charges are highly dependent so we remove minutes 
columnDrop = ['total eve minutes', 'total night minutes', 'total day minutes', 'total intl minutes']
#dropping the dependent featues
data_trainsdf.drop(columnDrop, axis =1 , inplace = True)
data_trainsdf.info()

testData_traindf = testdf.copy()
testData_traindf.drop(columnDrop, axis =1 , inplace = True)
testData_traindf.info()

#Now we seperate our results and data to train on
data_train = data_trainsdf.iloc[:,:-1].values
data_result = data_trainsdf.iloc[:,15].values

testData_train = testData_traindf.iloc[:,:-1].values
testData_result = testData_traindf.iloc[:,15].values


#As we have categorical data in our set we need to encode it to numbers so that it could fit in our models
labelEncoder_train = LabelEncoder()

#Encoding data_train & testData_train
#Encode states
data_train[:, 0] =labelEncoder_train.fit_transform(data_train[:, 0])
testData_train[:, 0] = labelEncoder_train.fit_transform(testData_train[:, 0])
#Encode international plan
data_train[:,3]  =labelEncoder_train.fit_transform(data_train[:, 3])
testData_train[:,3]  =labelEncoder_train.fit_transform(testData_train[:, 3])
#Encode voice mail plan
data_train[:,4] =labelEncoder_train.fit_transform(data_train[:, 4])
testData_train[:,4] =labelEncoder_train.fit_transform(testData_train[:, 4])


#Encoding data_result data
labelEncoder_result = LabelEncoder()
data_result = labelEncoder_result.fit_transform(data_result)
#Encoding testData_result 
testData_result = labelEncoder_result.fit_transform(testData_result)


#Feature Scaling
standardScaler = StandardScaler()
data_train = standardScaler.fit_transform(data_train)
testData_train = standardScaler.transform(testData_train)


#The data is processed now all we have to do is fit the model into the data 

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decisionTree.fit(data_train, data_result)


#predicting the testData_train 
predictedResultDecisionTree = decisionTree.predict(testData_train)


#Making a confusion matrix from decision tree classifier
confusionMatrixDecisionTree = confusion_matrix(testData_result, predictedResultDecisionTree)


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(n_estimators = 30, criterion = "entropy", random_state=0)
randomForest.fit(data_train, data_result)

#predicting the testData_train
predictedResultRandomForest = randomForest.predict(testData_train)


#Makifn a confusion matrix from random forest 
confusionMatrixRandomForest = confusion_matrix(testData_result, predictedResultRandomForest)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticClassifier = LogisticRegression()
logisticClassifier.fit(data_train, data_result)

#predicting the testData_train
predictedResultLogisticRegression = logisticClassifier.predict(testData_train)


#Making confusion matrix for logistic regression
confusionMatrixLogisticRegression = confusion_matrix(testData_result, predictedResultLogisticRegression)


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p = 2)
knnClassifier.fit(data_train, data_result)

#predicting the testData_train
predictedResultKNN = knnClassifier.predict(testData_train)


#Making the confusion matrix for knn classifier
confusionMatrixKNNClassifier = confusion_matrix(testData_result, predictedResultKNN)

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
naiveBClassifier = GaussianNB()
naiveBClassifier.fit(data_train, data_result)

#predictnig the testData_train
predictedResultNB = naiveBClassifier.predict(testData_train)


#Confusion Matrix for Naive Bayes Classifier
confusionMatrixNBClassifier = confusion_matrix(testData_result, predictedResultNB)


#SVM Classifier
from sklearn.svm import SVC
svmClassifier = SVC(kernel = 'rbf')
svmClassifier.fit(data_train, data_result)

#predicting the testData_train
predictedResultSVM = svmClassifier.predict(testData_train)


#Confusion Matrix for SVM
confusionMatrixSVM = confusion_matrix(testData_result, predictedResultSVM)


#Compiling all the confsuion matrix and making a nice plot to get the idea of all model
df_confusionMatrixDecisionTree= pd.DataFrame(confusionMatrixDecisionTree)
df_confusionMatrixRandomForest=pd.DataFrame(confusionMatrixRandomForest)
df_confusionMatrixLogisticRegression=pd.DataFrame(confusionMatrixLogisticRegression)
df_confusionMatrixKNNClassifier=pd.DataFrame(confusionMatrixKNNClassifier)
df_confusionMatrixNBClassifier=pd.DataFrame(confusionMatrixNBClassifier)
df_confusionMatrixSVM=pd.DataFrame(confusionMatrixSVM)


#Functions to check the specificity, recall, false positive rate, false negative rate
def specificity(cnList):
    FP=cnList[1,0]
    TN=cnList[1,1]
    return (TN*100)/(TN+FP)

def recall(cnList):
    TP=cnList[0,0]
    FN=cnList[0,1]
    return (TP*100)/(TP+FN)

def FPrate(cnList):
    FP=cnList[1,0]
    TN=cnList[1,1]
    return (FP*100)/(FP+TN)


def FNrate(cnList):
    TP=cnList[0,0]
    FN=cnList[0,1]
    return (FN*100)/(FN+TP)


print 'Accuracy Score of Decision Tree: ', accuracy_score(testData_result, predictedResultDecisionTree)*100,'%'
print 'Specificity of Decision Tree: ',specificity(confusionMatrixDecisionTree),'%'
print 'Recall of Decision Tree: ',recall(confusionMatrixDecisionTree),'%'
print 'False Positive Rate of Decision Tree: ',FPrate(confusionMatrixDecisionTree),'%'
print 'False Negative Rate of Decision Tree: ',FNrate(confusionMatrixDecisionTree),'%'
print '--------------->>>>>>>><<<<<<-------------'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of Decision Tree Classifier")
sn.heatmap(df_confusionMatrixDecisionTree, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])

print 'Accuracy Score of Random Forest :', accuracy_score(testData_result, predictedResultRandomForest)*100,'%'
print 'Specificity of Random Forest: ',specificity(confusionMatrixRandomForest),'%'
print 'Recall of Random Forest: ',recall(confusionMatrixRandomForest),'%'
print 'False Positive Rate of Random Forest: ',FPrate(confusionMatrixRandomForest),'%'
print 'False Negative Rate of Random Forest: ',FNrate(confusionMatrixRandomForest),'%'
print '--------------->>>>>>>><<<<<<-------------'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of Random Forest Classifier")
sn.heatmap(df_confusionMatrixRandomForest, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])

print 'Accuracy Score of Logistic Regression:', accuracy_score(testData_result, predictedResultLogisticRegression)*100,'%'
print 'Specificity of Logistic Regression: ',specificity(confusionMatrixLogisticRegression),'%'
print 'Recall of Logistic Regression: ',recall(confusionMatrixLogisticRegression),'%'
print 'False Positive Rate of Logistic Regression: ',FPrate(confusionMatrixLogisticRegression),'%'
print 'False Negative Rate of Logistic Regression: ',FNrate(confusionMatrixLogisticRegression),'%'
print '--------------->>>>>>>><<<<<<-------------'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of Logistic Regression")
sn.heatmap(df_confusionMatrixLogisticRegression, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])

print 'Accuracy Score of KNN Classifier:', accuracy_score(testData_result, predictedResultKNN)*100,'%'
print 'Specificity of KNN Classifier: ',specificity(confusionMatrixKNNClassifier),'%'
print 'Recall of KNN Classifier: ',recall(confusionMatrixKNNClassifier),'%'
print 'False Positive Rate of KNN Classifier: ',FPrate(confusionMatrixKNNClassifier),'%'
print 'False Negative Rate of KNN Classifier: ',FNrate(confusionMatrixKNNClassifier),'%'
print '--------------->>>>>>>><<<<<<-------------'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of KNN Classifier")
sn.heatmap(df_confusionMatrixKNNClassifier, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])

print 'Accuracy Score of Naive Bayes Classifier:', accuracy_score(testData_result, predictedResultNB)*100,'%'
print 'Specificity of Naive Bayes Classifier: ',specificity(confusionMatrixNBClassifier),'%'
print 'Recall of Naive Bayes Classifier: ',recall(confusionMatrixNBClassifier),'%'
print 'False Positive Rate of Naive Bayes Classifier: ',FPrate(confusionMatrixNBClassifier),'%'
print 'False Negative Rate of Naive Bayes Classifier: ',FNrate(confusionMatrixNBClassifier),'%'
print '--------------->>>>>>>><<<<<<-------------'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix for Naive Bayes Classifier")
sn.heatmap(df_confusionMatrixNBClassifier, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])

print 'Accuracy Score of SVM:', accuracy_score(testData_result, predictedResultSVM)*100,'%'
print 'Specificity of SVM: ',specificity(confusionMatrixSVM),'%'
print 'Recall of SVM: ',recall(confusionMatrixSVM),'%'
print 'False Positive Rate of SVM: ',FPrate(confusionMatrixSVM),'%'
print 'False Negative Rate of SVM: ',FNrate(confusionMatrixSVM),'%'
print '--------------->>>>>>>><<<<<<-------------'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix for SVM")
sn.heatmap(df_confusionMatrixSVM, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])

#Using a very popular and powerful algorithm XGBoost
xgbclassifier = XGBClassifier()
xgbclassifier.fit(data_train, data_result)

#predicting the testData_train
predictedResultXGB = xgbclassifier.predict(testData_train)

#Confusion Matrix for XGBoost
confusionMatrixXGB = confusion_matrix(testData_result, predictedResultXGB)

print 'Accuracy Score of XGB:', accuracy_score(testData_result, predictedResultXGB)*100,'%'
print 'Specificity of XGB: ',specificity(confusionMatrixXGB),'%'
print 'Recall of XGB: ',recall(confusionMatrixXGB),'%'
print 'False Positive Rate of XGB: ',FPrate(confusionMatrixXGB),'%'
print 'False Negative Rate of XGB: ',FNrate(confusionMatrixXGB),'%'
print '--------------->>>>>>>><<<<<<-------------'
df_confusionMatrixXGB=pd.DataFrame(confusionMatrixXGB)
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix for XGB")
sn.heatmap(df_confusionMatrixXGB, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])

