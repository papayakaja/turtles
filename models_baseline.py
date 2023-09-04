#!/usr/bin/env python
# coding: utf-8

# ## Baseline Model Notebook
# 
# Goals of this notebook:
# * Given y_true, y_pred, calculate the RMSE
# * Implement a basic evaluation function
# * Assuming we are given X_train, y_train, fit a basic model and evaluate it
# * Additionally, implement cross_validation scoring

# import libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import set_config
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression

set_config(transform_output="pandas")

# Set random seed 
RSEED = 42

#final_data = pd.read_csv("data/wrangled_data.csv")

# TODO: Solving date formate issue with date_caught
#del final_data['date_caught']

#final_data.info()

# Select X and y features
#X = final_data.drop(['capture_number'], axis = 1)
#y = final_data['capture_number']

# Splitting the dataset
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# Check the shape of the data sets
print("X_train:", X_train.shape)  
print("y_train:", y_train.shape)  
print("X_test:", X_test.shape) 
print("y_test:", y_test.shape)  

# Implement a basic evaluation function
def evaluate_rmse(y_true, y_pred, ndigits=3):
    """ Prints the RMSE (root mean squared error) of y_pred in relation to y_true"""
    rmse = mean_squared_error(y_true, y_pred, squared=False )
    print("Number of predictions: ", len(y_pred))
    print("RMSE: ", round(rmse, ndigits))
    return rmse

# Test the evaluation function
y_true_testing = [3, -0.5, 2, 7]
y_pred_testing = [2.5, 0.0, 2, 8]
#np.sqrt(sum((np.array(y_true_testing)-np.array(y_pred_testing))**2)/len(y_true_testing))
assert float(np.abs(evaluate_rmse(y_true_testing, y_pred_testing) - 0.612)) <= 0.001

# Assuming we are given X_train, y_train, fit a basic linear model and evaluate it
# initialize the model
lin_reg = LinearRegression()

# train model 
lin_reg.fit(X_train,y_train)

# make predictions on X_test
y_predicted = lin_reg.predict(X_test)

# evaluate
error = evaluate_rmse(y_test, y_predicted)

# Additionally, implement cross_validation scoring

scorer_rmse = make_scorer(mean_squared_error, squared=False)

#lr = LinearRegression()

print("CV RMSE scores: ", cross_val_score(lin_reg, X_train, y_train, cv=5, scoring=scorer_rmse, verbose=5))

# saving the model 
filename = 'models/linear_regression_model.sav'
pickle.dump(lin_reg, open(filename, 'wb'))

# ## Error analysis:
def error_analysis(y_test, y_pred):
    """Generated true vs. predicted values and residual scatter plot for models

    Args:
        y_test (array): true values for y_test
        y_pred_test (array): predicted values of model for y_test
    """     

    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    # Calculate residuals
    residuals = y_test["capture_number"] - y_pred[0]
    residuals = pd.DataFrame(residuals)
    
    # Plot real vs. predicted values 
    fig, ax = plt.subplots(1,2, figsize=(15, 5))
    plt.subplots_adjust(right=1)
    plt.suptitle('Error Analysis')
    
    ax[0].scatter(y_pred, y_test, color="#FF5A36", alpha=0.7)
    ax[0].plot([-400, 350], [-400, 350], color="#193251")
    ax[0].set_title("True vs. predicted values", fontsize=16)
    ax[0].set_xlabel("predicted values")
    ax[0].set_ylabel("true values")
    #ax[0].set_xlim((y_pred.min()-10), (y_pred.max()+10))
    #ax[0].set_ylim((y_test.min()-40), (y_test.max()+40))
    
    ax[1].scatter(y_pred, residuals, color="#FF5A36", alpha=0.7)
    ax[1].plot([-400, 350], [0,0], color="#193251")
    ax[1].set_title("Residual Scatter Plot", fontsize=16)
    ax[1].set_xlabel("predicted values")
    ax[1].set_ylabel("residuals")
    #ax[1].set_xlim((y_pred.min()-10), (y_pred.max()+10))
    #ax[1].set_ylim((residuals.min()-10), (residuals.max()+10));

error_analysis(y_test, y_predicted)