# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:33:43 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../Datasets/titanic/train.csv")

test_data = pd.read_csv("../Datasets/titanic/test.csv")


features = ["Pclass", "Sex", "SibSp", "Parch"]


X = data[features]
print(X.columns)
x = test_data[features]

#Since non-String values are required therefore replacing value of male with 1 and female with 2
for gender in X.Sex:
    if(gender == 'male'):
        X.Sex = 1
    else:
        X.Sex = 2


for gender in x.Sex:
    if(gender == 'male'):
        x.Sex = 1
    else:
        x.Sex = 2
        
Y = data.Survived
train_X,test_x,train_Y,test_y = train_test_split(X,Y,random_state=1)


model = RandomForestClassifier(n_estimators=10, max_features=1)
model.fit(train_X,train_Y)

        
predictions = model.predict(test_x)

print(mean_absolute_error(test_y,predictions))

model.fit(X,Y)

predictions = model.predict(x)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
