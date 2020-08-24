# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:02:04 2020

@author: Le Champion
"""


import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Importing dataset

data = pd.read_csv("../Datasets/housing_price/train.csv",index_col='Id')

test = pd.read_csv("../Datasets/housing_price/test.csv",index_col='Id')

Y = data.SalePrice
X = data.drop(['SalePrice'],axis=1)
T = test

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, Y, train_size=0.8,
                                            test_size=0.2,random_state=0)

# low cardinality categorical cols

obj_cols = [col for col in X_train_full.columns if X_train_full[col].nunique()<10
            and X_train_full[col].dtype == 'object']

# numeric cols

num_cols = [col for col in X_train_full.columns if X_train_full[col].dtype in 
            ['int64','float64']]

req_cols = obj_cols + num_cols

X_train = X_train_full[req_cols].copy()
X_valid = X_valid_full[req_cols].copy()
X_test = T[req_cols].copy()

# using one-hot encoding using pandas
# pd.get_dummies() : it convert categorical data into one-hot encoded data 

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

estimators = [100,200,400,500,600,850,1000,1250,1500]

estimators_2 = [4000]

for n_estimator in estimators_2:
    model = XGBRegressor(n_estimators = n_estimator,learning_rate = 0.05)
    model.fit(X_train,y_train,early_stopping_rounds=100, eval_set=[(X_valid, y_valid)],
             verbose=False)
    predictions = model.predict(X_test)

output = pd.DataFrame({'Id': test.index,'SalePrice': predictions})
output.to_csv('submission.csv', index=False)

    
    
    

