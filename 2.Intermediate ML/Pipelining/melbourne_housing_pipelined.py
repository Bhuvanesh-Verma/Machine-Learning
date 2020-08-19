# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:46:47 2020

@author: Le Champion
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

file_path = "../Datasets/Melbourne_Data_Set/melb_data.csv"
file_data = pd.read_csv(file_path)


#Chossing Dataset for model

y = file_data.Price
X = file_data.drop(['Price'],axis = 1)


#Creating training and testing data

train_x , test_x, train_y, test_y = train_test_split(X,y,random_state=1
                                                     ,train_size = 0.8,test_size=0.2)


# Select categorical columns
categorical_cols = [cname for cname in train_x.columns if train_x[cname].nunique() < 10 and 
                        train_x[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in train_x.columns if train_x[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = train_x[my_cols].copy()
X_valid = train_x[my_cols].copy()


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
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=0)
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(train_x, train_y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(test_x)

# Evaluate the model
score = mean_absolute_error(test_y, preds)
print('MAE:', score)