# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:25:54 2020

@author: Le Champion
"""


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


filepath = "../Dataset/insurance.csv"
data = pd.read_csv(filepath)

plt.figure(figsize=(10,6))

print(data)

sns.swarmplot(x=data['smoker'],y=data['charges'])
