# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:50:48 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("../Datasets/Melbourne_Data_Set/melb_data.csv")

#setting target variable
y = data.Price

#Considering all columns except for Price as features, we were filter out
# required features in next steps
X = data.drop(['Price'],axis = 1)


#splitting data into 80% training and 20% testing
train_x,test_x,train_y,test_y = train_test_split(X,y,train_size = 0.8,test_size=0.2)

#removing columns with null value from training and testing dataset

cols_with_missing_values = [col for col in train_x.columns if train_x[col].isnull().any()]

train_x.drop(cols_with_missing_values,axis =1 ,inplace = True)
test_x.drop(cols_with_missing_values,axis =1 ,inplace = True)

#In order to implement label encoding we need to find variables with low cardinality
# Cardinality is number of unique values in a column

cols_with_low_card = [col for col in train_x.columns if train_x[col].nunique() < 10
                      and train_x[col].dtype == "object"]

# choosing columns with numerical values
cols_with_num_values = [col for col in train_x.columns if train_x[col].dtype 
                        in ['int64','float64']]

#keeping only selected columns as feature

cols = cols_with_low_card + cols_with_num_values
train_X = train_x[cols].copy()
test_X = test_x[cols].copy()

#Getting Categorical variables
s = (train_X.dtypes == 'object')

object_cols = list(s[s].index)

#One-Hot Encoding

encoder = OneHotEncoder(sparse=False,handle_unknown= 'ignore')

#encoder will apply one-hot encoding technique on categorical variables
one_hot_train_cols = pd.DataFrame(encoder.fit_transform(train_X[object_cols]))
one_hot_test_cols = pd.DataFrame(encoder.transform(test_X[object_cols]))

#index was removed while encoding, therefore putting it back
one_hot_train_cols.index = train_X.index
one_hot_test_cols.index = test_X.index

#now we dont need categorical columns

temp_train = train_X.drop(object_cols,axis =1)
temp_test = test_X.drop(object_cols,axis=1)

#finally creating one-hot encoded dataset

one_hot_train_x = pd.concat([temp_train,one_hot_train_cols],axis =1)
one_hot_test_x = pd.concat([temp_test,one_hot_test_cols],axis =1)



model = RandomForestRegressor(n_estimators=100, random_state=0) # defining a model

#fitting model
model.fit(one_hot_train_x,train_y) 


#predicting values
predicted_values = model.predict(one_hot_test_x)

print(mean_absolute_error(test_y,predicted_values))

