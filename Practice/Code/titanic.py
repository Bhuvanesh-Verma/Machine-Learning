# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:58:33 2020

@author: Le Champion
"""

def updateUsingMap(data_point):
    '''
    This function make data more readable by changing 
    Survived columns values to Yes if 1 
    and No if 0
    Parameters
    ----------
    data_point : int

    Returns
    -------
    data_point : string

    '''
    updated_data = ''
    if(data_point==0):
        updated_data = "No"
    else:
        updated_data = "Yes"
    return updated_data
        
def updateUsingApply(row):
    '''
    This function make data more readable by changing 
    Survived columns values to Yes if 1 
    and No if 0
    Parameters
    ----------
    row : Series

    Returns
    -------
    row : Series

    '''

    print(type(row))
    if(row.Survived==0):
        row.Survived = "No"
    else:
        row.Survived = "Yes"
    return row

import pandas as pd

pd.set_option('display.max_columns', 11)

data = pd.read_csv("../Dataset/titanic.csv")
test = data
test.Sex.replace("male","M",inplace=True)
test.Sex.replace("female","F",inplace=True)
test.rename(columns={"Sex":"Gender"},inplace=True)
data.rename_axis("Sr.No",axis='rows',inplace=True)
data.rename_axis("Catergory",axis='columns',inplace=True)
print(test.head(2))