# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:04:52 2020

@author: Le Champion
"""


import pandas as pd

# Reading a csv file in pandas
data = pd.read_csv("../Dataset/wine_review.csv",index_col=0)

# Using head() to get top values and tail() to get bottom values

top_6_rows = data.head(6)

last_4_rows = data.tail(4)

# Using describe() to know statistical values for numerical data

stats = data.describe()

price_stat = data.price.describe()

price_mean = data.price.mean() # you can also use .median(), .sum() for median and sum.

unique_countries = data.country.unique()

occurrence_for_each_country = data.country.value_counts()

