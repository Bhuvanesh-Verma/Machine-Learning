# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:07:54 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
    model = RandomForestRegressor(max_leaf_nodes=400,random_state=1)
    model.fit(trainX,trainY)
    predictions = model.predict(x)
    return predictions


data = pd.read_csv("../Datasets/housing_price/train.csv",index_col='Id')

test = pd.read_csv("../Datasets/housing_price/test.csv",index_col='Id')
T = test 
Y = data.SalePrice
data.drop(['SalePrice'],axis=1,inplace=True)
X =data

#Electrical column is removed since it has NA value in training data set only
# All other columns with NA will be imputed
T.drop(['Electrical'],axis=1,inplace=True)
X.drop(['Electrical'],axis=1,inplace=True)

# find column with missing data

cols_with_train_missing_data = [col for col in X.columns if X[col].isnull().any()]

cols_with_test_missing_data = [col for col in T.columns if T[col].isnull().any()]


uncommon_na_cols = set(cols_with_test_missing_data)-set(cols_with_train_missing_data)
common_na_cols = set(cols_with_train_missing_data)-set(cols_with_test_missing_data)

# I have choosen to keep only those na_columns which are common in both training and testing 
# dataset. Testing contain all na_columns which are in training. So I can impute all those
# columns but for this model I will pick common na_columns


X_copy =X.copy()
X_copy.drop(uncommon_na_cols,axis=1)
T_copy = T.copy()
T_copy.drop(uncommon_na_cols,axis=1)



#Lets use Imputation Extension to deal with missing data
cols_for_na = []
numeric_cols = []

for col in cols_with_train_missing_data:
    if X_copy[col].dtype in ['int64','float64']:
        numeric_cols.append(col)
        name = col + "_was_missing"
        X_copy[name] = X_copy[col].isnull()
        T_copy[name] = T_copy[col].isnull()
        cols_for_na.append(name)

imputer = SimpleImputer()
imputed_train_data = pd.DataFrame(imputer.fit_transform(X_copy[numeric_cols]))
imputed_test_data = pd.DataFrame(imputer.transform(T_copy[numeric_cols]))


imputed_train_data.columns = X_copy[numeric_cols].columns
imputed_test_data.columns = T_copy[numeric_cols].columns

X_copy.drop(numeric_cols,axis=1)
T_copy.drop(numeric_cols,axis=1)



X_copy = pd.concat([X_copy,imputed_train_data],axis=1)
T_copy = pd.concat([X_copy,imputed_test_data],axis=1)
print(X_copy)

# Lets deal with categorical variables

#First retrieve all categorical variables except for _was_missing variables

temp = (X_copy.dtypes == 'object')
temp1 = (T_copy.dtypes == 'object')
print(set(temp) == set(temp1))
object_cols = temp[temp].index



