# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:00:34 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

file_path = "../Datasets/Melbourne_Data_Set/melb_data.csv"
file_data = pd.read_csv(file_path)


#Chossing Dataset for model

y = file_data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = file_data[features]

#Creating training and testing data

train_x , test_x, train_y, test_y = train_test_split(X,y,random_state=1)
for i in [10,100,500,5000]:
    print(get_mae(i,train_x,test_x,train_y,test_y))
    
    

