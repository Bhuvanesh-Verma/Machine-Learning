# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:07:54 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
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

# find column with missing data in dataset


cols_with_num_missing_data_T = [col for col in T.columns if T[col].dtype in 
                ['int64','float64'] and T[col].isnull().any()]
cols_with_num_missing_data_X = [col for col in X.columns if X[col].dtype in 
                ['int64','float64'] and X[col].isnull().any()]


# remove missing data with string entries
cols_with_object_missing_data_T= [col for col in T.columns if T[col].dtype =='object'
                                 and T[col].isnull().any()]

cols_with_object_missing_data_X = [col for col in X.columns if X[col].dtype =='object'
                                 and X[col].isnull().any()]
cols_to_remove = cols_with_object_missing_data_T + cols_with_object_missing_data_X

X_copy =X.copy()
X_copy = X_copy.drop(cols_to_remove,axis=1)

T_copy = T.copy()
T_copy = T_copy.drop(cols_to_remove,axis=1)


#Lets use Imputation Extension to deal with missing data
cols_for_na = []
numeric_na_cols = cols_with_num_missing_data_T + cols_with_num_missing_data_X



for col in numeric_na_cols:
    name = col + "_was_missing"
    X_copy[name] = X_copy[col].isnull()
    T_copy[name] = T_copy[col].isnull()
    cols_for_na.append(name)
    

imputer = SimpleImputer()
imputed_train_data = pd.DataFrame(imputer.fit_transform(X_copy[numeric_na_cols]))
imputed_test_data = pd.DataFrame(imputer.transform(T_copy[numeric_na_cols]))


imputed_train_data.index = X_copy.index
imputed_test_data.index = T_copy.index

imputed_train_data.columns = X_copy[numeric_na_cols].columns
imputed_test_data.columns = T_copy[numeric_na_cols].columns

X_copy = X_copy.drop(numeric_na_cols,axis=1)
T_copy = T_copy.drop(numeric_na_cols,axis=1)


X_copy = pd.concat([X_copy,imputed_train_data],axis=1)
T_copy = pd.concat([T_copy,imputed_test_data],axis=1)



# Lets deal with categorical variables

#First retrieve all categorical variables except for _was_missing variables

temp = (X_copy.dtypes == 'object')
temp1 = (T_copy.dtypes == 'object')
object_cols = temp[temp].index

# We will use one-hot encoding to only those columns with less than 10 cardinality
unique_objects = list(map(lambda col: X_copy[col].nunique(),object_cols))
temp = dict(zip(object_cols,unique_objects))

low_card_cols = [col for col in object_cols if X_copy[col].nunique() <10]
high_card_cols = list(set(object_cols) - set(low_card_cols))



# Encoding with One-Host Encoder
encoder = OneHotEncoder(sparse=False,handle_unknown= 'ignore')
temp_train = X_copy.drop(high_card_cols,axis=1)
temp_valid = T_copy.drop(high_card_cols,axis=1)

one_hot_train_cols = pd.DataFrame(encoder.fit_transform(X_copy[low_card_cols]))
one_hot_test_cols = pd.DataFrame(encoder.transform(T_copy[low_card_cols]))

one_hot_train_cols.index = X_copy.index
one_hot_test_cols.index = T_copy.index

temp_train = temp_train.drop(low_card_cols,axis=1)
temp_valid = temp_valid.drop(low_card_cols,axis=1)

OH_X_train = pd.concat([temp_train,one_hot_train_cols],axis =1)
OH_X_valid = pd.concat([temp_valid,one_hot_test_cols],axis=1)

pred = calcPredictions(OH_X_train,Y,OH_X_valid)


output = pd.DataFrame({'Id': test.index,'SalePrice': pred})
output.to_csv('submission.csv', index=False)
