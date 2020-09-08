# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:36:21 2020

@author: Le Champion
"""



import pandas as pd

data = pd.read_csv("../Dataset/wine_review.csv",index_col=0)

# rename() : to rename column name and index

data.rename(columns={'price':'cost'},inplace=True); # changing column name

data.rename(index={0:'One',1:'Two',2:'Three'},inplace=True) # changing index

# rename_axis() : To give name to row and column axis
data.rename_axis("Sr.No",axis='rows',inplace=True)
data.rename_axis("Catergory",axis='columns',inplace=True)


print(data)