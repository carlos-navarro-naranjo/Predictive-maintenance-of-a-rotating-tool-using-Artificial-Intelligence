# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 19:20:21 2022

@author: richm
"""
import nltk
import sklearn
import scipy
print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The scipy  version is {}.'.format(scipy.__version__))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as m


# *** THIS IMPORTS YOUR FUNCTION ***
#from a4util import encode_one_hot

# Import pandas package
#import pandas as pd

import numpy as np
data= pd.read_csv("predictive_maintenance.csv")

def encode_one_hot  (data):
 results= np.zeros((10000,3))
 y= data.loc[:,"Type"]

 for i in range(10000):
  if y[i] == 'M': 
      results[i,0]=1
  elif y[i]== 'L': 
      results[i,1]=1
  else: 
     results[i,2]=1
 
 print(y)
 print(results)

 data.insert(1, "L", results[:,0], True)
 data.insert(2, "M", results[:,1], True)
 data.insert(3, "H", results[:,2], True)
 data.drop('Type', inplace=True, axis=1)

 print(data)

 return data



     


# This will work if the CSV is in the same folder. Otherwise, edit to 
# add the appropriate path to the file.
df = encode_one_hot(pd.read_csv("predictive_maintenance.csv"))

features = ['L', 
            'M', 
            'H', 
            'Air temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'] 


response = 'Target'

X = df[features]
y = df[response]

n_pos = y.sum()
n_neg = y.size - n_pos


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.8, 
                                                    random_state=999)


# info about datasets
print('\nDataset Statistics\t\tTraining\t  Test\t\t  Total')
print('\tPositive examples :\t %5d \t\t %5d \t\t %5d' % 
      (y_train.sum(), y_test.sum(), y.sum()))
print('\tNegative examples :\t %5d \t\t %5d \t\t %5d' %
      (y_train.size - y_train.sum(), y_test.size - y_test.sum(), y.size - y.sum()))
print('\tTotal             :\t %5d \t\t %5d \t\t %5d' % 
      (y_train.size, y_test.size, y.size))



# *** perfect on training set
# # this defines a pipeline with a scaler and classifier
est = [('scaler', StandardScaler()),
            ('clf', MLPClassifier((100),
                                  activation='tanh',
                                  solver='lbfgs',
                                  alpha=0.0001,
                                  max_iter=1000,
                                  random_state=42,
                                  tol=1e-4))]



# construct the pipeline and fit the model
pipe = Pipeline(est)
pipe.fit(X_train, y_train)


# evaluation
train_pred = pipe.predict(X_train)
test_pred = pipe.predict(X_test)
print('\n\nFITTING REPORT')
print('Metric\t\t\t\t\t\tTraining\t  Test')
print('\taccuracy            :    %6.4f\t\t %6.4f' % 
      (m.accuracy_score(y_train, train_pred), 
       m.accuracy_score(y_test, test_pred)))
      
print('\tbalanced accuracy   :    %6.4f\t\t %6.4f' % 
      (m.balanced_accuracy_score(y_train, train_pred),
       m.balanced_accuracy_score(y_test, test_pred)))

print('\tprecision           :    %6.4f\t\t %6.4f' % 
      (m.precision_score(y_train, train_pred),
       m.precision_score(y_test, test_pred)))

print('\trecall              :    %6.4f\t\t %6.4f' % 
      (m.recall_score(y_train, train_pred),
       m.recall_score(y_test, test_pred)))

print('\tF1 score            :    %6.4f\t\t %6.4f' % 
      (m.f1_score(y_train, train_pred),
       m.f1_score(y_test, test_pred)))

print('\tROC AUC score       :    %6.4f\t\t %6.4f' % 
      (m.roc_auc_score(y_train, train_pred),
       m.roc_auc_score(y_test, test_pred)))


disp= m.ConfusionMatrixDisplay.from_predictions(pipe, X_test, y_test, 
                                        display_labels=('No Failure', 'Failure'))
