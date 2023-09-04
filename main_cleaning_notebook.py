#!/usr/bin/env python
# coding: utf-8


import seaborn as sns
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn import set_config
set_config(transform_output="pandas")

# Set random seed 
RSEED = 42

warnings.filterwarnings("ignore")

# import the csv file
df = pd.read_csv("data/data.csv")

#df.head()

variable_def = pd.read_csv("data/variable_definitions.csv",encoding='latin-1')
pd.set_option('display.max_colwidth', None)
#variable_def

#df.info()

#df.isnull().sum()

# no duplicates of the rescue IDs
#df.nunique()


# ## First data Cleaning

#replace uppercase letters with lowercase in column names
df.columns = df.columns.str.lower()

#format column names
df = df.rename({'date_timecaught':'date_caught', 'capturesite':'capture_site', 'foragingground':'foraging_ground',
                'capturemethod':'capture_method', 'landingsite':'landing_site', 'turtlecharacteristics':'turtle_characteristics',
                'releasesite':'release_site', 'date_timerelease':'date_released',},axis=1)

# Dropping not needed columns
df.drop(["rescue_id", "fisher", "researcher", "sex","turtle_characteristics", "tag_1", "lost_tags", "t_number"], axis=1, inplace=True)

#convert date column to datetime type
import datetime
df['date_caught'] = pd.to_datetime(df['date_caught'])
df['date_released'] = pd.to_datetime(df['date_released'], errors='coerce')

# converting all entries into lower case to get rid of "Creek" and "creek"
df["foraging_ground"] = df["foraging_ground"].apply(lambda x: x.lower())

# change the types to 0 and 1, "ocean" = 1, "creek" = 0
df["foraging_ground"] = df["foraging_ground"].apply(lambda x: 1 if x == "ocean" else 0)

df['tag_2'].fillna(0, inplace=True) 
# Replacing string values in Tag_2 column (which represent a large turtle) with 1:
df['tag_2'] = df['tag_2'].replace(to_replace='.*', value=1, regex=True)

#df['tag_2'].unique()
#df['tag_2'].value_counts()

#Impute NaN CCL_cm values, setting all of them as median
ccl_cm_median = df['ccl_cm'].median()
df['ccl_cm'].fillna(ccl_cm_median, inplace=True) 
#df.isnull().sum()

# change to a bool to take up less memory
df["foraging_ground"].astype(bool)

df["capture_method"] = df["capture_method"].apply(lambda x: x.lower())

#df["landing_site"].unique()

# One-hot encode the 'features' data using pandas.get_dummies()
categorical_features = ["capture_method", "foraging_ground", "landing_site", "species"]
df = pd.get_dummies(df,columns = categorical_features)
#df.head()

#fill ccw_cm NaNs with mean
ccw_mean = df['ccw_cm'].mean()
df['ccw_cm'].fillna(ccw_mean, inplace=True)

#fill weight NaNs with mode (8.5)
df['weight_kg'].fillna(8.5, inplace=True)

#convert status to category
df['status'] = df['status'].astype('category')
df['release_site'] = df['release_site'].astype('category')
df["capture_site"] = df["capture_site"].astype("category")
#fill release_site NaNs with mode and convert to category
df['release_site'].fillna(df['release_site'].mode()[0], inplace=True)

#convert status & release_site to numeric with label encoder
df['status'] = LabelEncoder().fit_transform(df['status'])
df['release_site'] = LabelEncoder().fit_transform(df['release_site'])

# check on missing values, only left for date released
#df.isna().sum() 

#fill weight NaNs with mode (8.5)
df['date_released'].fillna(0, inplace=True)

# clean up column names
df.columns = df.columns.str.replace("landing_site_LandingSite_CaptureSiteC","cs_c")
df.columns = df.columns.str.replace("species_S","s")
df.columns = df.columns.str.replace("capture_method","cm")

# export data to csv, index to True 
df.to_csv('data/cleaned_data.csv', index=True)