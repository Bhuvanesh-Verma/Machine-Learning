# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:39:20 2020

@author: Le Champion
"""


import pandas as pd

# Reading a csv file in pandas
data = pd.read_csv("../Dataset/wine_review.csv",index_col=0)

# Selecting data in column-first , row-second manner

countries = data.country # or data['country']

#Selecting a particular data in a column

country1 = data["country"][5] # or data.country[5]


#Selecting data in row-first column-second manner using loc and iloc

row1 = data.loc[0] # or data.iloc[0]

# Selecting particular data in a row

country = data.loc[0,"country"] # or data.iloc[0,0]

# Selecting a series of rows from data

rows = data.loc[:9] # or data.iloc[:10] => returns first 10 rows(0-9)

#Selecting series of rows for a particular column/s

countries = data.loc[2:7,"country"] # or data.iloc[2:8,0]

countries_with_points = data.loc[:5,['country','points']] # to use iloc here we need
# index of "points" column which is possible to find but loc provides a better way

# Selecting particular rows

specific_rows = data.loc[[0,4,5,1,8]] # or data.iloc[[0,4,5,1,8]]

# Selecting data with condition

high_points_wine = data.loc[data.points >= 95]  # or data[data.points >=95]

#Assigning values to a column

data['quality'] = range(0,len(data))

# We can also set index to a column which we find is suitable 

data.set_index("quality")

print(data)
