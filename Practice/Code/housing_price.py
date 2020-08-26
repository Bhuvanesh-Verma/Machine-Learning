# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:52:21 2020

@author: Le Champion
"""


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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


    
data = pd.read_csv("../Dataset/housing_price/train.csv",index_col='Id')

test = pd.read_csv("../Dataset/housing_price/test.csv",index_col='Id')

T = test 
Y = data.SalePrice
data.drop(['SalePrice'],axis=1,inplace=True)
X =data
features = ["PoolArea","TotRmsAbvGrd","2ndFlrSF","1stFlrSF",
            "YearBuilt","YearRemodAdd","LotArea","MSSubClass","OpenPorchSF","Fireplaces",
            "GrLivArea","YrSold","3SsnPorch","WoodDeckSF","HalfBath","FullBath",
        "EnclosedPorch","OverallQual","OverallCond", ]


#  *******************************NUMERIC**DATA*************************************

#Dealing with numeric data
num_cols_to_impute = ['LotFrontage','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                   'TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath',
                    'GarageCars', 'GarageArea']


#  ***************************CATEGORICAL**DATA*****************************

#Dealing with Categorical Data

cols = [col for col in T.columns if T[col].dtype == 'object'
        and not( T[col].isnull().any())]

# training object variables with no NA 
train_not_na_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 
                      'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
                      'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                      'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                      'ExterQual', 'ExterCond', 'Foundation', 'Heating', 
                      'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 
                      'PavedDrive', 'SaleType', 'SaleCondition']
 
#test object variable with no NA
test_not_na_cols = ['Street', 'LotShape', 'LandContour', 'LotConfig', 
                     'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                     'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
                     'ExterQual', 'ExterCond', 'Foundation', 'Heating', 
                     'HeatingQC', 'CentralAir', 'Electrical', 'PavedDrive', 
                     'SaleCondition']



# common objects in both dataset with no na
non_na = list(set(train_not_na_cols) and set(test_not_na_cols))



# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols_to_impute),
        ('cat', categorical_transformer, non_na)
    ])

model = XGBRegressor(n_estimators = 900,learning_rate = 0.05)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X,Y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(test)




output = pd.DataFrame({'Id': test.index,'SalePrice': preds})
output.to_csv('submission.csv', index=False)