# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:21:46 2020

@author: Le Champion
"""

import pandas as pd

#Series is singular component of a Dataframe. Series is just a list.
# Dataframe is like a table. It is collection of multiple series.

# Series


index = pd.Series(["One","Two"])
colors = pd.Series(['Blue','Green'],index = index) 
names = pd.Series(['Ram','Shyam'],index = index)

# Creating a Dataframe
data = pd.DataFrame({"Colors":colors,"Names":names},index=index)

#Saving a csv

data.to_csv("basic.csv")