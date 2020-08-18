# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:25:10 2020

@author: Le Champion
"""



import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def calcError(trainX,trainY,x):
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
    




data = pd.read_csv("../Datasets/housing_price/train.csv")

test = pd.read_csv("../Datasets/housing_price/test.csv")

y = data.SalePrice

#Columns with non-String value is choosen in features.
features = ["PoolArea","TotRmsAbvGrd","2ndFlrSF","1stFlrSF",
            "YearBuilt","YearRemodAdd","LotArea","MSSubClass","OpenPorchSF","Fireplaces",
            "GrLivArea","YrSold","3SsnPorch","WoodDeckSF","HalfBath","FullBath",
        "EnclosedPorch","OverallQual","OverallCond", 'BsmtFinSF1',
      'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
        'BsmtHalfBath']


X = data[features]

x = test[features]
#print(X.columns[X.isna().any()].tolist())  # to check if X contain any NaN columns

train_X,test_x,train_Y,test_y = train_test_split(X,y,random_state=1)


#working with missed data

#3 Imputation Extension

x_copy = x.copy()
cols_with_missing = [col for col in x.columns
                     if x[col].isnull().any()]
train_X_copy = train_X.copy()

for col in cols_with_missing:
    x_copy[col + "_was_missing"] = x_copy[col].isnull()
    train_X_copy[col + "_was_missing"] = train_X_copy[col].isnull()
   

my_imputer = SimpleImputer()

imputed_test_data = pd.DataFrame(my_imputer.fit_transform(x_copy))

imputed_test_data.columns = x_copy.columns

prediction_list = calcError(train_X_copy,train_Y,imputed_test_data)


output = pd.DataFrame({'Id': test.Id,'SalePrice': prediction_list})
output.to_csv('submission.csv', index=False)

