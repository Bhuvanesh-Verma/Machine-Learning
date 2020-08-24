# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:12:31 2020

@author: Le Champion
"""


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# Paths of the files to read
iris_set_filepath = "../Dataset/iris_setosa.csv"
iris_ver_filepath = "../Dataset/iris_versicolor.csv"
iris_vir_filepath = "../Dataset/iris_virginica.csv"

# Read the files into variables 
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

# Change the style of the figure to the "dark" theme
sns.set_style("ticks")

plt.figure(figsize=(10,8))
# Histograms for each species
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()