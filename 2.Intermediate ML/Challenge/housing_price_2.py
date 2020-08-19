# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:32:05 2020

@author: Le Champion
"""


import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def calcPredictions(trainX,trainY,x):
    '''
    This function use RandomForestRegressor to build a model using
    training data train_X and train_Y. max_leaf_nodes parameter can be adjusted 
    to get better results. I have tried with 10,100,400,500,700 and 400
    gave best results. After fitting model we predict results using x dataset.
    Parameters
    ----------
    train_X : dataset for training.
    train_Y : target SalePrice for training .
    x : dataset for testing.

    Returns
    -------
    predictions : predictions array.

    '''
    model = RandomForestRegressor(max_leaf_nodes=400,random_state=1)
    model.fit(trainX,trainY)
    predictions = model.predict(x)
    return predictions

def encodingOrdinalLabel(df):
    '''
    This functions uses predefined maps for labelling values in provided dataframes.
    After labeling, original columns are replaced with new labeled columns
    Parameters
    ----------
    df : Dataframe
        This dataframe is used to create Oridinal Labels It contains actual data,
        which needs to be labeled.

    Returns
    -------
    object_df : Dataframe
        Labeled dataframe is returned.

    '''
    condition_map = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
    slope_map = {'Gtl':1,'Mod':2,'Sev':3}
    paved_map = {'Y':3,'P':2,'N':1}
    lost_shape_map = {'Reg':4,"IR1":3,"IR2":2,"IR3":1}
    
    object_df = pd.DataFrame(df)
    object_df['temp_exter_cond'] = object_df.ExterCond.map(condition_map)
    object_df['temp_land_slope'] = object_df.LandSlope.map(slope_map)
    object_df['temp_paved_drive'] = object_df.PavedDrive.map(paved_map)
    object_df['temp_lot_shape'] = object_df.LotShape.map(lost_shape_map)
    object_df['temp_exter_qual'] = object_df.ExterQual.map(condition_map)
    object_df['temp_heating_qc'] = object_df.HeatingQC.map(condition_map)
    
    temp = ['temp_exter_cond','temp_land_slope','temp_paved_drive','temp_lot_shape',
            'temp_exter_qual','temp_heating_qc']
    object_df = object_df.drop(cols_for_label,axis=1)
    object_df['ExterCond'] = object_df['temp_exter_cond'] 
    object_df['LandSlope'] = object_df['temp_land_slope']
    object_df['PavedDrive'] = object_df['temp_paved_drive']
    object_df['LotShape'] = object_df['temp_lot_shape']
    object_df['ExterQual'] = object_df['temp_exter_qual']
    object_df['HeatingQC'] = object_df['temp_heating_qc']
    
    object_df = object_df.drop(temp,axis=1)
    return object_df
    
data = pd.read_csv("../Datasets/housing_price/train.csv",index_col='Id')

test = pd.read_csv("../Datasets/housing_price/test.csv",index_col='Id')

T = test 
Y = data.SalePrice
data.drop(['SalePrice'],axis=1,inplace=True)
X =data
features = ["PoolArea","TotRmsAbvGrd","2ndFlrSF","1stFlrSF",
            "YearBuilt","YearRemodAdd","LotArea","MSSubClass","OpenPorchSF","Fireplaces",
            "GrLivArea","YrSold","3SsnPorch","WoodDeckSF","HalfBath","FullBath",
        "EnclosedPorch","OverallQual","OverallCond", ]


#  *******************************NUMERIC**DATA*************************************

#Dealing with numeric data
num_cols_to_impute = ['LotFrontage','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                   'TotalBsmtSF', 'BsmtFullBath','BsmtHalfBath',
                    'GarageCars', 'GarageArea']

total_features = features + num_cols_to_impute
X_copy = X[total_features]
T_copy = T[total_features]

for col in num_cols_to_impute:
    name = col + "_was_missing"
    X_copy[name] = X_copy[col].isnull()
    T_copy[name] = T_copy[col].isnull()

imputer = SimpleImputer()
imputed_num_train_data = pd.DataFrame(imputer.fit_transform(
    X_copy[num_cols_to_impute]))
imputed_num_test_data = pd.DataFrame(imputer.fit_transform(
    T_copy[num_cols_to_impute]))
imputed_num_test_data.index = T.index
imputed_num_train_data.index = X.index

imputed_num_train_data.columns = X_copy[num_cols_to_impute].columns
imputed_num_test_data.columns = T_copy[num_cols_to_impute].columns

X_copy = X_copy.drop(num_cols_to_impute,axis = 1)
T_copy = T_copy.drop(num_cols_to_impute,axis = 1)

new_X = pd.concat([X_copy,imputed_num_train_data],axis=1)
new_T = pd.concat([T_copy,imputed_num_test_data],axis=1)

#  ***************************CATEGORICAL**DATA*****************************

#Dealing with Categorical Data

cols = [col for col in T.columns if T[col].dtype == 'object'
        and not( T[col].isnull().any())]

# training object variables with no NA 
train_not_na_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 
                      'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
                      'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                      'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                      'ExterQual', 'ExterCond', 'Foundation', 'Heating', 
                      'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 
                      'PavedDrive', 'SaleType', 'SaleCondition']
 
#test object variable with no NA
test_not_na_cols = ['Street', 'LotShape', 'LandContour', 'LotConfig', 
                     'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                     'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
                     'ExterQual', 'ExterCond', 'Foundation', 'Heating', 
                     'HeatingQC', 'CentralAir', 'Electrical', 'PavedDrive', 
                     'SaleCondition']

# common objects in both dataset with no na
non_na = list(set(train_not_na_cols) and set(test_not_na_cols))


non_na.remove('Electrical') # includes NA data
non_na.remove('Neighborhood') # too many variable for Encoding

cols_for_label = ['ExterCond','LandSlope','PavedDrive','LotShape','ExterQual',
                  'HeatingQC']
cols_for_OH = list(set(non_na) - set(cols_for_label))

# creating maps for label encoding

train_df = encodingOrdinalLabel(X[cols_for_label])
test_df = encodingOrdinalLabel(T[cols_for_label])

new_X = pd.concat([new_X,train_df],axis=1)
new_T = pd.concat([new_T,test_df],axis=1)

# We will use one-hot encoding to only those columns with less than 10 cardinality
unique_objects = list(map(lambda col: X[col].nunique(),cols_for_OH))
temp = dict(zip(cols_for_OH,unique_objects))
print(sorted(temp.items(),key=lambda x : x[1])) #it shows all cols in cols_for_OH
# have cardinality less than 10


# Encoding with One-Host Encoder
encoder = OneHotEncoder(sparse=False,handle_unknown= 'ignore')
one_hot_train_cols = pd.DataFrame(encoder.fit_transform(X[cols_for_OH]))
one_hot_test_cols = pd.DataFrame(encoder.transform(T[cols_for_OH]))

one_hot_train_cols.index = X.index
one_hot_test_cols.index = T.index


new_X = pd.concat([new_X,one_hot_train_cols],axis=1)
new_T = pd.concat([new_T,one_hot_test_cols],axis=1)

pred = calcPredictions(new_X,Y,new_T)

output = pd.DataFrame({'Id': test.index,'SalePrice': pred})
output.to_csv('submission.csv', index=False)