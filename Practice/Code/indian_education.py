# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:12:20 2020

@author: Le Champion
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../Dataset/indian_education/dropout-ratio-2012-2015.csv")

data.Primary_Boys=data.Primary_Boys.str.replace('NR','0')
data.Primary_Boys = data.Primary_Boys.astype('float')
print(data.Primary_Boys.value_counts())
sns.heatmap(data.Primary_Boys,annot=True)