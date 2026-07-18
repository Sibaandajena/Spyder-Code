# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:08:36 2026
Topic - Logistic regression with future prediction
@author: SIBA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv(r'C:\Users\HP\Desktop\ALL DATA SET\logit classification.csv')


X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=20, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l1', solver='liblinear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test) 


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test, y_pred)
print(ac)


from sklearn.metrics import  classification_report
cr= classification_report(y_test, y_pred)
print(cr)


bias = classifier.score(X_train, y_train)
print(bias)

variance = classifier.score(X_test, y_test)
print(variance)


#---------------------FEATURE PREDICTIO-----------------------


dataset1= pd.read_csv(r'C:\Users\HP\Desktop\ALL DATA SET\Future prediction1.csv')

d2= dataset1.copy()

dataset1= dataset1.iloc[:, [2, 3]].values

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
M= sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()


d2['y_pred1']= classifier.predict(M)

d2.to_csv('final1.csv')


'''
#  to get the path
import os
os.getcwd()
'''

# ******************************************************

from sklearn.metrics import roc_auc_score, roc_curve
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

auc_score= roc_auc_score(y_test, y_pred_prob)
print(auc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.Figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC= {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--') # random classifier line
plt.xlabel1('False Positive Rate')
plt.ylabel1('True Positive Rate')
plt.title('RDC Curve')
plt.legend(loc='Lower right')
plt.grid()
plt.show()

