# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:38:16 2020

@author: Le Champion
"""


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


filepath = "../Dataset/flight_delays.csv"
data = pd.read_csv(filepath, index_col="Month")


# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=data, annot=True)
