# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:18:08 2020

@author: Le Champion
"""


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

data_loc = '../Dataset/fifa.csv'
data = pd.read_csv(data_loc,index_col='Date',parse_dates = True)

plt.figure(figsize=(17,6))
sns.lineplot(data=data['ARG'])