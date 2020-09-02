# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:12:09 2020

@author: Le Champion
"""


import pandas as pd

data = pd.read_csv("../Dataset/wine_review.csv")

all_datatypes_in_data = data.dtypes

points_datatype = data.points.dtype

# to change datatype of a column

changed_points = data.points.astype('str')
print(changed_points)
