# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:33:43 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def calcPredictions(n,d):
    
    model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                         ])
    
    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, Y)
    
    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)
    
    return preds
    



data = pd.read_csv("../Datasets/titanic/train.csv")

test_data = pd.read_csv("../Datasets/titanic/test.csv")

T =test_data.copy()
Y = data.Survived

X =data.drop(['Survived'],axis=1)


train_x , test_x, train_y, test_y = train_test_split(X,Y,random_state=1
                                                     ,train_size = 0.8,test_size=0.2)



# Select numeric columns

num_cols = [col for col in X.columns if X[col].dtype in ['int64','float64']]

# Select object columns

obj_cols = [col for col in X.columns if X[col].dtype == 'object']


# Keep selected columns only

num_cols.remove('PassengerId')
my_cols = ['Sex']+ num_cols
X_train = X[my_cols].copy()
X_valid = T[my_cols].copy()



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
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, ['Sex'])
    ])

pred = calcPredictions(100,5)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
output.to_csv('my_submission.csv', index=False)















