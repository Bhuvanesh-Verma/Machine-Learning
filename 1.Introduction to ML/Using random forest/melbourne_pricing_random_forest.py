# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:00:34 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor


file_path = "../Datasets/Melbourne_Data_Set/melb_data.csv"
file_data = pd.read_csv(file_path)


#Chossing Dataset for model

y = file_data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = file_data[features]

#Creating training and testing data

train_x , test_x, train_y, test_y = train_test_split(X,y,random_state=1)

model = RandomForestRegressor(random_state=1)
model.fit(train_x,train_y)
predictions = model.predict(test_x)

print(mean_absolute_error(test_y,predictions))
    
    

