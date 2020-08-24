# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:24:59 2020

@author: Le Champion
"""




import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

data_loc = '../Dataset/spotify.csv'
data = pd.read_csv(data_loc,index_col='Date',parse_dates = True)

plt.figure(figsize=(17,6))
print(list(data.columns))
#plt.title("Spotify Listener for Something Just Like This By Coldplay ft. Chainsmokers")
sns.lineplot(data=data['Shape of You'])