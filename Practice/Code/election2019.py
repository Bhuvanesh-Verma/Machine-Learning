# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:36:16 2020

@author: Le Champion
"""


import pandas as pd

data = pd.read_csv("../Dataset/ls2019.csv")
data["SYMBOL"].fillna("Unknown",inplace=True)
data["GENDER"].fillna("Unknown",inplace=True)
data["CRIMINAL\nCASES"].fillna("Unknown",inplace=True)
data["AGE"].fillna("Unknown",inplace=True)
data["CATEGORY"].fillna("Unknown",inplace=True)
data["EDUCATION"].fillna("Unknown",inplace=True)
data["ASSETS"].fillna("Unknown",inplace=True)
data["LIABILITIES"].fillna("Unknown",inplace=True)

