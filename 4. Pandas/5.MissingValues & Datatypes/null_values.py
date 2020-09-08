# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:26:44 2020

@author: Le Champion
"""


import pandas as pd

data = pd.read_csv("../Dataset/wine_review.csv",index_col=0)

# Missing data is represeted with NaN i.e Not a Number

na_in_country = pd.isnull(data.country) # it returns a series of Boolean values
na_in_country = data[na_in_country] # all rows with boolean value True is retured

# Filling data with na values

updated_country = data.country.fillna("Unknown")

# Replacing non null values

updated_us_name = data.country.replace("US","United States of America")
