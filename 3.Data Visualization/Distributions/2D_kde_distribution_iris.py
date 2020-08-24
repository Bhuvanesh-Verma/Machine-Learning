# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:07:52 2020

@author: Le Champion
"""




import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns




# Path of the file to read
iris_filepath = "../Dataset/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")
# Change the style of the figure to the "dark" theme
sns.set_style("white")

# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")