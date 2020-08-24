# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:34:38 2020

@author: Le Champion
"""


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


filepath = "../Dataset/flight_delays.csv"
data = pd.read_csv(filepath, index_col="Month")

plt.figure(figsize=(10,6))

plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")


sns.barplot(x=data.index, y=data['NK'])

plt.ylabel("Arrival delay (in minutes)")