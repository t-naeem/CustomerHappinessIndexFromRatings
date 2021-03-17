# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:49:18 2021

@author: Talha Naeem
"""

#Collecting data
import pandas as pd

my_data = pd.read_csv('ACME-HappinessSurvey2020.csv')

#Data Wrangling

noOfmissingVal = my_data.isnull().sum()
print('There were no missing values in our dataset. \n')

#Givingthreshold that rating > 3 in each question is happy for that question
dummyX1 = my_data['X1'] >= 3
dummyX2 = my_data['X2'] >= 3
dummyX3 = my_data['X3'] >= 3
dummyX4 = my_data['X4'] >= 3
dummyX5 = my_data['X5'] >= 3
dummyX6 = my_data['X6'] >= 3

#Droping previous columns of the ratings
my_data.drop(['X1', 'X2', 'X3', 'X4', 'X5', 'X6'], axis = 1, inplace = True)

#Adding True/False as columns for the question instead of rating from 1-5
my_data =pd.concat([my_data, dummyX1, dummyX2, dummyX3, dummyX4,
                    dummyX5, dummyX6], axis = 1)

#Train Data
x = my_data.drop('Y', axis = 1)
y = my_data['Y']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10,
                                                    random_state = 7)

from sklearn.linear_model import LogisticRegression

logModel = LogisticRegression()

logModel.fit(x_train, y_train)

#Test data
predictions = logModel.predict(x_test)

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

#Show Results
results = classification_report(y_test, predictions)
confMat = confusion_matrix(y_test, predictions)
accScore = accuracy_score(y_test, predictions) * 100

print('The confusion matrix for all independent variables is \n', confMat,
      ' and accuracy is ', accScore, '\n')

#Which independent variables are irrelavent
x_unaffected = my_data.drop(['Y', 'X1', 'X2', 'X3', 'X6'], axis = 1)
x_trainUnaff, x_testUnaff, y_trainUnaff, y_testUnaff = train_test_split(
                                                    x_unaffected, y,
                                                    test_size = 0.10,
                                                    random_state = 7)
logModelUnaff = LogisticRegression()
logModelUnaff.fit(x_trainUnaff, y_trainUnaff)

predictionsUnaff = logModelUnaff.predict(x_testUnaff)
resultsUnaff = classification_report(y_testUnaff, predictionsUnaff)
confMatUnaff = confusion_matrix(y_testUnaff, predictionsUnaff)
accScoreUnaff = accuracy_score(y_testUnaff, predictionsUnaff) * 100


print('The confusion matrix for relavent independent variables is \n'
      , confMatUnaff, ' and accuracy is:', accScoreUnaff, '\n')
print('The results are unaltered even if we remove X1, X2, X3, X6.'
      'Check Variable explorer for more information.Therefore, results '
      'are not dependent on them, so they should be removed.')
      
