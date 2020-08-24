# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:33:43 2020

@author: Le Champion
"""


import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def calcPredictions(trainX,trainY,x):
    '''
    This function use RandomForestRegressor to build a model using
    training data train_X and train_Y. max_leaf_nodes parameter can be adjusted 
    to get better results. I have tried with 10,100,400,500,700 and 400
    gave best results. After fitting model we predict results using x dataset.
    Parameters
    ----------
    train_X : dataset for training.
    train_Y : target SalePrice for training .
    x : dataset for testing.

    Returns
    -------
    predictions : predictions array.

    '''
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(trainX,trainY)
    predictions = model.predict(x)
    return predictions



data = pd.read_csv("../Datasets/titanic/train.csv")

test_data = pd.read_csv("../Datasets/titanic/test.csv")

T =test_data.copy()
Y = data.Survived

X =data.drop(['Survived'],axis=1)

# Select numeric columns

num_cols = [col for col in X.columns if X[col].dtype in ['int64','float64']]

# Select object columns

obj_cols = [col for col in X.columns if X[col].dtype == 'object']

# Select num columns with NAN data

na_num_cols_X = [col for col in num_cols if X[col].isnull().any()]
na_num_cols_T = [col for col in num_cols if T[col].isnull().any()]

na_num_cols = list(set(na_num_cols_T).union(set(na_num_cols_X)))

# Impute NAN columns

X_copy = X[num_cols].copy()
T_copy = T[num_cols].copy()

for col in na_num_cols:
    name = col + "_was_missing"
    X_copy[name] = X_copy[col].isnull()
    T_copy[name] = T_copy[col].isnull()

imputer = SimpleImputer()
imputed_num_train_data = pd.DataFrame(imputer.fit_transform(
    X_copy[na_num_cols]))
imputed_num_test_data = pd.DataFrame(imputer.fit_transform(
    T_copy[na_num_cols]))
imputed_num_test_data.index = T.index
imputed_num_train_data.index = X.index

imputed_num_train_data.columns = X_copy[na_num_cols].columns
imputed_num_test_data.columns = T_copy[na_num_cols].columns

X_copy = X_copy.drop(na_num_cols,axis = 1)
T_copy = T_copy.drop(na_num_cols,axis = 1)

new_X = pd.concat([X_copy,imputed_num_train_data],axis=1)
new_T = pd.concat([T_copy,imputed_num_test_data],axis=1)


# Lets deal with Categorical Data


# Encoding with One-Host Encoder
encoder = OneHotEncoder(sparse=False,handle_unknown= 'ignore')
one_hot_train_cols = pd.DataFrame(encoder.fit_transform(X[['Sex']]))
one_hot_test_cols = pd.DataFrame(encoder.transform(T[['Sex']]))

one_hot_train_cols.index = X.index
one_hot_test_cols.index = T.index


new_X = pd.concat([new_X,one_hot_train_cols],axis=1)
new_T = pd.concat([new_T,one_hot_test_cols],axis=1)


pred = calcPredictions(new_X,Y,new_T)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
output.to_csv('my_submission.csv', index=False)


















