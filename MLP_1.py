# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:55:03 2022

@author: Titan
"""

# coding: utf-8


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sys
import gzip
import shutil
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import seaborn as sns
sns.set_theme(style="ticks", color_codes=True)



pathtr1 = r'C:\Users\Titan\OneDrive - Aston University\Desktop\PIXNET\Thesis Osaka\paper_pr\Training_data.csv'  #experimental data
pathtr2 = r'C:\Users\Titan\OneDrive - Aston University\Desktop\PIXNET\Thesis Osaka\paper_pr\Trainings_data.csv'  #simulation data
pathtr2 = r'C:\Users\Titan\OneDrive - Aston University\Desktop\PIXNET\Thesis Osaka\paper_pr\Training_data_35_50.csv'


training = pd.read_csv(pathtr1)
training.columns = ['f1','f2','f3','symbol','power','gain']
training = training[(training.power>=35)&(training.power<=40)]
print(training.describe().transpose())

#Extraction of amplitude values
Xt = training.iloc[:,[0,1,2]]

#extraction of tags
y_train = training.iloc[:,3]
#y_validation = validation.iloc[:,4]

#rows and columns count
y_train = y_train.astype({'symbol':int})

#data division for training and test

trainX, testX, trainY, testY = train_test_split(Xt, y_train, test_size = 0.3, random_state=(1))


sns.catplot(x="symbol", y="f1", data=training)
sns.catplot(x="symbol", y="f2", data=training)
sns.catplot(x="symbol", y="f3", data=training)

sc=StandardScaler()
sc.fit(trainX)
trainX_scaled = sc.transform(trainX)
testX_scaled = sc.transform(testX)

mlp_clf = MLPClassifier()


print('Trainign = Rows: %d, columns: %d' % (trainX.shape[0], trainX.shape[1]))
print('Test = Rows: %d, columns: %d' % (testX.shape[0], testX.shape[1]))

# best result with no filters 6,12,24,48  26/07/2022 MLP_1_CVtest.pkl
param_grid = {
    'hidden_layer_sizes': [(12,18,36) ],
    'max_iter': [6000],
    'activation': ['relu'],
    'solver': ['sgd'],
}

grid = GridSearchCV(mlp_clf, param_grid, cv=5,verbose=3)
grid.fit(trainX_scaled, trainY)
y_pred = grid.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

print('Best Parameters: ', grid.best_params_) 

joblib_file2 = "MLP_1_CV_11.pkl"  


joblib.dump(grid.best_estimator_, joblib_file2)
cm= confusion_matrix(testY, y_pred)


### experimental models

#MLP_1_CV_1.pkl   has data from -6 to 50 dB    Training_data.csv    (6,12,18,36)  Accuracy: 0.60
#MLP_1_CV_2.pkl   has data from -6 to 50 dB    Training_data.csv    (6,12,18) Accuracy: 0.59
#MLP_1_CV_3.pkl   has data from -6 to 50 dB    Training_data.csv    (12,18,36) Accuracy: 0.60
#MLP_1_CV_4.pkl   has data from -6 to 50 dB    Training_data.csv    (6,12,24) Accuracy: 0.59

#MLP_1_CV_5.pkl   has data from 0 to 50 dB    Training_data.csv    (12,18,36)  Accuracy: 0.66
#MLP_1_CV_6.pkl   has data from 10 to 50 dB    Training_data.csv    (12,18,36) Accuracy: 0.79
#MLP_1_CV_7.pkl   has data from 15 to 50 dB    Training_data.csv    (12,18,36) Accuracy: 0.87
#MLP_1_CV_8.pkl   has data from 20 to 50 dB    Training_data.csv    (12,18,36)  Accuracy: 0.94

#MLP_1_CV_9.pkl   has data from 25 to 30 dB    Training_data.csv    (12,18,36)  Accuracy: 0.92
#MLP_1_CV_10.pkl   has data from 30 to 35 dB    Training_data.csv    (12,18,36) Accuracy: 0.98
#MLP_1_CV_11.pkl   has data from 35 to 40 dB    Training_data.csv    (12,18,36) 

############# simulation models
#MLP_1_CV_1s.pkl   has data from -6 to 50 dB    Trainings_data.csv    (6,12,18,36)  Accuracy: 0.78
#MLP_1_CV_2s.pkl   has data from -6 to 50 dB    Trainings_data.csv    (6,12,18) Accuracy: 0.77
#MLP_1_CV_3s.pkl   has data from -6 to 50 dB    Trainings_data.csv    (12,18,36) Accuracy: 0.78
#MLP_1_CV_4s.pkl   has data from -6 to 50 dB    Trainings_data.csv    (6,12,24) Accuracy: 0.77

#MLP_1_CV_5s.pkl   has data from 0 to 50 dB    Trainings_data.csv    (12,18,36)  Accuracy: 0.83
#MLP_1_CV_6s.pkl   has data from 10 to 50 dB    Trainings_data.csv    (12,18,36) Accuracy: 0.94
#MLP_1_CV_7s.pkl   has data from 15 to 50 dB    Trainings_data.csv    (12,18,36) Accuracy: 0.99
#MLP_1_CV_8s.pkl   has data from 20 to 50 dB    Trainings_data.csv    (12,18,36) Accuracy: 0.99

#MLP_1_CV_9s.pkl   has data from 25 to 30 dB    Trainings_data.csv    (12,18,36) Accuracy: 1.00
