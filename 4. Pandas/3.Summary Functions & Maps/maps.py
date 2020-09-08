# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:35:56 2020

@author: Le Champion
"""

def increasePrice(price):
    return price+100

def increasePriceUsingApply(row):
    row.price = row.price+100
    return row
import pandas as pd

# Reading a csv file in pandas
data = pd.read_csv("../Dataset/wine_review.csv",index_col=0)


# map function : For each iteration map() supplies a value as parameter for the function supplied 
# and returns the manipulated value. It does it for each row of specified column
# and finally returns a new series with manipulated values.
new_price = data.price.map(increasePrice)

# apply function : It functions almost same as map but it supplies rows to given function
# which in return provide manipulated row and finally after all iteration of rows
# it returns a new Dataframe

new_price_using_apply = data.apply(increasePriceUsingApply,axis='columns')

