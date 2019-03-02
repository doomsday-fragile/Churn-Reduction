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

#Now we seperate our results and data to train on
data_train = df.iloc[:,:-1].values
data_result = df.iloc[:,19].values

#Same for the test data 
testData_train = testdf.iloc[:,:-1].values
testData_result = testdf.iloc[:,19].values


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
testData_train = standardScaler.fit_transform(testData_train)


#The data is processed now all we have to do is fit the model into the data 

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decisionTree.fit(data_train, data_result)


#predicting the testData_train 
predictedResultDecisionTree = decisionTree.predict(testData_train)


#Making a confusion matrix from decision tree classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusionMatrixDecisionTree = confusion_matrix(testData_result, predictedResultDecisionTree)
print 'Accuracy Score :', accuracy_score(testData_result, predictedResultDecisionTree)*100,'%'

print("Confusion Matrix of Decision Tree Classifier")
print(confusionMatrixDecisionTree)


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
randomForest.fit(data_train, data_result)

#predicting the testData_train
predictedResultRandomForest = randomForest.predict(testData_train)


#Makifn a confusion matrix from random forest 
confusionMatrixRandomForest = confusion_matrix(testData_result, predictedResultRandomForest)
print("Confusion Matrix of Random Forest Classifier")
print(confusionMatrixRandomForest)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logisticClassifier = LogisticRegression()
logisticClassifier.fit(data_train, data_result)

#predicting the testData_train
predictedResultLogisticRegression = logisticClassifier.predict(testData_train)


#Making confusion matrix for logistic regression
confusionMatrixLogisticRegression = confusion_matrix(testData_result, predictedResultLogisticRegression)
print("Confusion Matrix of Logistic Regression")
print(confusionMatrixLogisticRegression)


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p = 2)
knnClassifier.fit(data_train, data_result)

#predicting the testData_train
predictedResultKNN = knnClassifier.predict(testData_train)


#Making the confusion matrix for knn classifier
confusionMatrixKNNClassifier = confusion_matrix(testData_result, predictedResultKNN)
print("Confusion Matrix of KNN Classifier")
print(confusionMatrixKNNClassifier)


#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
naiveBClassifier = GaussianNB()
naiveBClassifier.fit(data_train, data_result)

#predictnig the testData_train
predictedResultNB = naiveBClassifier.predict(testData_train)


#Confusion Matrix for Naive Bayes Classifier
confusionMatrixNBClassifier = confusion_matrix(testData_result, predictedResultNB)
print("Confusion Matrix for Naive Bayes Classifier")
print(confusionMatrixNBClassifier)


#SVM Classifier
from sklearn.svm import SVC
svmClassifier = SVC(kernel = 'linear')
svmClassifier.fit(data_train, data_result)

#predicting the testData_train
predictedResultSVM = svmClassifier.predict(testData_train)


#Confusion Matrix for SVM
confusionMatrixSVM = confusion_matrix(testData_result, predictedResultSVM)
print("Confusion Matrix for SVM")
print(confusionMatrixSVM)


#Compiling all the confsuion matrix and making a nice plot to get the idea of all model

df_confusionMatrixDecisionTree= pd.DataFrame(confusionMatrixDecisionTree)
df_confusionMatrixRandomForest=pd.DataFrame(confusionMatrixRandomForest)
df_confusionMatrixLogisticRegression=pd.DataFrame(confusionMatrixLogisticRegression)
df_confusionMatrixKNNClassifier=pd.DataFrame(confusionMatrixKNNClassifier)
df_confusionMatrixNBClassifier=pd.DataFrame(confusionMatrixNBClassifier)
df_confusionMatrixSVM=pd.DataFrame(confusionMatrixSVM)


print 'Accuracy Score of Decision Tree :', accuracy_score(testData_result, predictedResultDecisionTree)*100,'%'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of Decision Tree Classifier")
sn.heatmap(df_confusionMatrixDecisionTree, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])
plt.savefig('decsionTree.png', bbox_inches = 'tight')

print 'Accuracy Score of Random Forest :', accuracy_score(testData_result, predictedResultRandomForest)*100,'%'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of Random Forest Classifier")
sn.heatmap(df_confusionMatrixRandomForest, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])
plt.savefig('randomforest.png', bbox_inches = 'tight')

print 'Accuracy Score of Logistic Regression :', accuracy_score(testData_result, predictedResultLogisticRegression)*100,'%'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of Logistic Regression")
sn.heatmap(df_confusionMatrixLogisticRegression, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])
plt.savefig('logisitcRegression.png', bbox_inches = 'tight')

print 'Accuracy Score of KNN Classifier :', accuracy_score(testData_result, predictedResultKNN)*100,'%'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix of KNN Classifier")
sn.heatmap(df_confusionMatrixKNNClassifier, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])
plt.savefig('knn.png', bbox_inches = 'tight')

print 'Accuracy Score of Naive Bayes Classifier :', accuracy_score(testData_result, predictedResultNB)*100,'%'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix for Naive Bayes Classifier")
sn.heatmap(df_confusionMatrixNBClassifier, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])
plt.savefig('nb.png', bbox_inches = 'tight')

print 'Accuracy Score of SVM :', accuracy_score(testData_result, predictedResultSVM)*100,'%'
plt.figure(figsize = (2,2))
plt.title("Confusion Matrix for SVM")
sn.heatmap(df_confusionMatrixSVM, annot=True, fmt='.0f', xticklabels = ['Yes','No'],yticklabels=['Yes','No'])
plt.savefig('svm.png', bbox_inches = 'tight')




#Making plot to show the importance of each feture using XGBoost library
from sklearn.ensemble import ExtraTreesClassifier
#Making a dmatrix for featrue importance scaling
df = df.drop(['Churn'], axis =1)
feature_names = list(df)
Xtrain = pd.DataFrame(data_train, columns = df.columns)
forest = ExtraTreesClassifier(n_estimators = 250)
forest.fit(data_train, data_result)
importance =forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)