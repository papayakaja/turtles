#!/usr/bin/env python
# coding: utf-8

import seaborn as sns

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from sklearn import set_config
set_config(transform_output="pandas")

# Set random seed 
RSEED = 42

warnings.filterwarnings("ignore")

df = pd.read_csv("data/cleaned_data.csv")

capture_site_category = pd.read_csv('data/CaptureSite_category.csv')

pd.set_option('display.max_colwidth', None)

#df.columns

#capture_site_category.head()

#format column names
capture_site_category = capture_site_category.rename({'CaptureSite':'capture_site','CaptureSiteCategory':'cs_category','Type':'type'},axis=1)

#capture_site_category.info()

categorical_columns = ['capture_site', 'cs_category', 'type']

# convert to categories
for col in categorical_columns:
    capture_site_category[col] = capture_site_category[col].astype('category')

df["date_caught"] = pd.to_datetime(df["date_caught"])
df["year"] = df["date_caught"].dt.year
df["week_of_year"] = df["date_caught"].dt.isocalendar().week  # Using isocalendar() method
df["year_woy"] = df["year"] * 100 + df["week_of_year"]
#df.head()

#df.columns

# Extracting the target variable from the dataset
target=df.groupby(["year_woy","capture_site"]).capture_site.count().rename("capture_number").reset_index()

#target.info()

#target.isnull().sum()

#capture_site_category.head()

# merging the tables
df_1 = pd.merge (left=df, right=capture_site_category, left_on ='capture_site', right_on = 'capture_site', how='left')

final_data=df_1.merge(target,on=["year_woy","capture_site"],how="left")

#final_data.isnull().sum()

#final_data['capture_number'].value_counts()

#final_data.info()

#final_data['capture_site'].nunique()

final_data['capture_site'] = LabelEncoder().fit_transform(final_data['capture_site'])

#final_data['cs_category'].nunique()

final_data['cs_category'] = LabelEncoder().fit_transform(final_data['cs_category'])

#final_data['type'].nunique()

final_data['type'] = LabelEncoder().fit_transform(final_data['type'])

del final_data['date_released'] 

final_data.drop(columns="date_caught", inplace=True)

#final_data['capture_number']

# Select X and y features
X = final_data.drop(['capture_number'], axis = 1)
y = final_data['capture_number']

#X.info()

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

# Check the shape of the data sets
print("X_train:", X_train.shape)  
print("y_train:", y_train.shape)  
print("X_test:", X_test.shape) 
print("y_test:", y_test.shape)   

#final_data['capture_number'].value_counts()

#final_data.info()

# export data to csv, index to True 
final_data.to_csv('data/wrangled_data.csv', index=True)

# saving the test data
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
