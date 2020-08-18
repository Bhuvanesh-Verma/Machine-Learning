# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:43:06 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

result = {}
data = pd.read_csv("../Datasets/housing_price/train.csv",index_col='Id')

test = pd.read_csv("../Datasets/housing_price/test.csv",index_col='Id')
X_test = test 



y = data.SalePrice
data.drop(['SalePrice'],axis=1,inplace=True)
X =data



# remove column with missing data

cols_with_missing_data = [col for col in X.columns if X[col].isnull().any()]

X.drop(cols_with_missing_data,axis=1,inplace=True)
X_test.drop(cols_with_missing_data,axis=1,inplace=True)

#split data into training and testing
train_X,test_x,train_Y,test_y = train_test_split(X,y,random_state=1,
                                                 train_size = 0.8,test_size=0.2)


# get categorical variables

temp = (train_X.dtypes == 'object')
object_cols = temp[temp].index

# Method 1: Dropping Categorical Variables

num_train_data = train_X.drop(object_cols,axis=1)

num_test_data = test_x.drop(object_cols,axis =1)

method_1_predictions = calcPredictions(num_train_data,train_Y,num_test_data)
result["Method 1"] = mean_absolute_error(test_y,method_1_predictions)


#Method 2: Label Encoding

# check if  training set columns value and testing set columns value for categorical
#   variables are same. If not same remove that column otherwise error is generated.

cols_to_remove = [col for col in object_cols 
                  if set(train_X[col].unique()) != set(test_x[col].unique())]

cols_to_keep = list(set(object_cols)-set(cols_to_remove))

label_train_X = train_X.drop(cols_to_remove,axis =1)
label_test_x = test_x.drop(cols_to_remove,axis=1)



# Apply label encoder 
encoder = LabelEncoder()

for col in cols_to_keep:
    label_train_X[col] = encoder.fit_transform(train_X[col])
    label_test_x[col] = encoder.transform(test_x[col])


method_2_predictions = calcPredictions(label_train_X,train_Y,label_test_x)
result["Method 2"] = mean_absolute_error(test_y,method_2_predictions)


#Method 3: One-Hot Encoding

# map(function,iterables) => map function takes parameter for function from iterable object
# lambda x : x+5 => lambda is a function which takes multiple parameters but single enpression
# Here we take column from object_cols list and count unique values 
unique_objects = list(map(lambda col: train_X[col].nunique(),object_cols))


# map(iterable,iterable, ...) => it joins iterable objects
# a=(1,2,3); b = (4,5,6); map(a,b); ((1,4),(2,5),(3,6))

temp = dict(zip(object_cols,unique_objects))

# We will use one-hot encoding to only those columns with less than 10 cardinality

low_card_cols = [col for col in object_cols if train_X[col].nunique() <10]
high_card_cols = list(set(object_cols) - set(low_card_cols))


# Encoding with One-Host Encoder
encoder = OneHotEncoder(sparse=False,handle_unknown= 'ignore')
temp_train = train_X.drop(high_card_cols,axis=1)
temp_valid = test_x.drop(high_card_cols,axis=1)

one_hot_train_cols = pd.DataFrame(encoder.fit_transform(train_X[low_card_cols]))
one_hot_test_cols = pd.DataFrame(encoder.transform(test_x[low_card_cols]))

one_hot_train_cols.index = train_X.index
one_hot_test_cols.index = test_x.index

temp_train = temp_train.drop(low_card_cols,axis=1)
temp_valid = temp_valid.drop(low_card_cols,axis=1)

OH_X_train = pd.concat([temp_train,one_hot_train_cols],axis =1)
OH_X_valid = pd.concat([temp_valid,one_hot_test_cols],axis=1) 

method_3_predictions = calcPredictions(OH_X_train,train_Y,OH_X_valid)
result["Method 3"] = mean_absolute_error(test_y,method_3_predictions)

