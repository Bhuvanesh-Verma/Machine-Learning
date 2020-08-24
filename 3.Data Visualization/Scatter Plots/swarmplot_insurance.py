# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:27:29 2020

@author: Le Champion
"""



import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


filepath = "../Dataset/insurance.csv"
data = pd.read_csv(filepath)

plt.figure(figsize=(16,6))

print(data)

sns.lmplot(x="bmi", y="charges", hue="smoker", data=data)