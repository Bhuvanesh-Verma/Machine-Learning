# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:08:39 2020

@author: Le Champion
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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


#Label Encoding : Choosing columns for labelling

s = (train_X.dtypes == 'object')

object_cols = list(s[s].index)

train_x_label = train_X.copy()
test_x_label = test_X.copy()

encoder = LabelEncoder()


#encoder randomly assign unique integer for unique values
for col in object_cols:
    train_x_label[col] = encoder.fit_transform(train_X[col])
    test_x_label[col] = encoder.transform(test_X[col])
    
model = RandomForestRegressor(n_estimators=100, random_state=0) # defining a model

#fitting model
model.fit(train_x_label,train_y) 


#predicting values
predicted_values = model.predict(test_x_label)

print(mean_absolute_error(test_y,predicted_values))
