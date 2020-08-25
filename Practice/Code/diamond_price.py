# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:42:18 2020

@author: Le Champion
"""


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor 

file_path = '../Dataset/diamonds.csv'
data = pd.read_csv(file_path,index_col=0)
data.index.name = 'Id'

print(data)
#sns.scatterplot(x=data['carat'],y=data['price'])
 

#sns.barplot(x=data['cut'],y=data['carat'])

#sns.distplot(a=data['carat'], kde=False)

#sns.kdeplot(data=data['depth'], shade=True)

#sns.jointplot(x=data['depth'], y=data['table'], kind="kde")

X = data.drop(['price','Id'],axis=1)

#features = ['carat','depth','table','x','y','z']
Y = data.price
#X=X[features].copy()


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,test_size=0.2)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# DecisionTreeRegressor mae=1026.992 with numeric cols only
# DecisionTreeRegressor mae=358.9 with one-hot encoded

model = XGBRegressor(n_estimators=500,learning_rate=0.1)
model.fit(X_train,Y_train)
predictions=model.predict(X_test)
print(mean_absolute_error(Y_test,predictions))



