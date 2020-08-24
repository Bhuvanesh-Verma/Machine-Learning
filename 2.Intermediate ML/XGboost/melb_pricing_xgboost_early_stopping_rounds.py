# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:48:47 2020

@author: Le Champion
"""


import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Read the data
data = pd.read_csv("../Datasets/Melbourne_Data_Set/melb_data.csv")

# Select features

features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']

X = data[features]
Y = data.Price

X_train, X_valid, y_train, y_valid = train_test_split(X, Y)

model = XGBRegressor(n_estimators = 400)
model.fit(X_train,y_train,early_stopping_rounds=7, eval_set=[(X_valid, y_valid)],
             verbose=False)
predictions = model.predict(X_valid)
print(mean_absolute_error(y_valid,predictions))