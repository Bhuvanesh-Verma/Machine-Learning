# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:44:58 2020

@author: Le Champion
"""


import pandas as pd

CAdata = pd.read_csv("../Dataset/CAvideos.csv",index_col=0)
INdata = pd.read_csv("../Dataset/CAvideos.csv",index_col=0)


# concat() : both dataset contains same fields, so concat will combine them to form
# a single dataset
CA_INdata = pd.concat([CAdata,INdata])

# join() : combines data based on common index

left = CAdata.set_index(['title','trending_date'])
right = INdata.set_index(['title','trending_date'])

joined_data = left.join(right,lsuffix="_CAN",rsuffix="_IND") # suffix is required since both
# dataset have same column names




