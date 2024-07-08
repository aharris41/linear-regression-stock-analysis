#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:03:56 2023

@author: alexandra
"""

""""
MAD3703 Project
Goal of project: To analyze a set of data using linear regression in order to predict the best linear
equation to predict the dependent variables based on the independent variables. Create a possible neural net
that will find one of the following:
    - best home to purchase/show up in general when given parameters such as home location, price, etc. 
    - best stock to buy depending on time of year, growth of company/its gross, etc, parameters surrounding life, etc
    
    
FINAL IDEA: We are doing linear regression to see how the open price, lowest price, highest price, and volume influence the closing...
... price of the stock. This will be done on every stock in the data set

*** POSSIBLY ANALYZE THE ENTIRE YEAR/FINAL PRICE, but then you need to get the other data set and make your own from it because
.. that one is too long
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv as csv
from sklearn.model_selection import train_test_split # testing and training data
from sklearn.linear_model import LinearRegression # also self explanatory
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler # for scaling data 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
import mplfinance as mpf # for fancy stock graph

# i lied we're doing linear regression on one file only i hate this

# Functions
"""
def convert_to_numeric_allcols(dataframe, column_index): # Converts all columns in a given data frame to numeric
    columns = dataframe.columns[column_index]
    dataframe[columns] = pd.to_numeric(dataframe[columns], errors = 'coerce')
"""
"""
def convert_to_numeric_allcols(dataframe):  # Converts all columns in a given data frame to numeric
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    dataframe.loc[:, numeric_cols] = dataframe.loc[:, numeric_cols].apply(pd.to_numeric, errors='coerce')
"""
"""
def convert_to_numeric_allcols(dataframe):
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    return dataframe
"""
def convert_to_numeric_except(dataframe, exclude_columns):
    columns_to_convert = [col for col in dataframe.columns if col not in exclude_columns]
    dataframe[columns_to_convert] = dataframe[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    return dataframe

def format_scientific(num, precision=3):
    if num == 0:
        return '0.0'
    else:
        exponent = int(np.log10(abs(num)))
        mantissa = num / (10 ** exponent)
        formatted_num = f"{mantissa:.{precision}f} × 10^{exponent}"
        return formatted_num
    
# Opening file
psfile = f'/Users/alexandra/Desktop/MAD3703 Project/new york stock exchange/prices-split-adjusted.csv'
fFile = f'/Users/alexandra/Desktop/MAD3703 Project/new york stock exchange/fundamentals.csv'

file1 = open(psfile)
file2 = open(fFile)

# Creating the data frames
stock_df = pd.read_csv(file1)
fundamentals_df = pd.read_csv(file2)
#print(stock_df) # good

# Dealing with fundamentals_df to extract the columns i want to use
chosen_columns = ['Ticker Symbol', 'Period Ending', 'Profit Margin', 'Gross Margin', 'Operating Margin', 'Total Revenue']
subset_df = fundamentals_df[chosen_columns]
print(subset_df) # good for all

# Start and end dates for subset_df to filter out
start_date1 = "2010-12-31"
end_date1 = "2016-12-31"

# Converting date to float date
subset_df['Period Ending'] = pd.to_datetime(subset_df['Period Ending'])
subset_df = subset_df[(subset_df['Period Ending'] >= start_date1) & (subset_df['Period Ending'] <= end_date1)]
subset_df['float Period Ending'] = subset_df['Period Ending'].dt.year * 10000 + subset_df['Period Ending'].dt.month * 100 + subset_df['Period Ending'].dt.day 

# Grouping by stock name
subset_df_sorted = subset_df.sort_values(by='Ticker Symbol')
#print(subset_df_sorted) # good

# Separate the dataframe by stock name, because we will be analyzing each stock individually and looking at the interesting ones
stock_names = stock_df.groupby('symbol')
#print(stock_names)

# Converting the date column to something more reasonable
stock_df['date'] = pd.to_datetime(stock_df['date'])
stock_df['float date'] = stock_df['date'].dt.year * 10000 + stock_df['date'].dt.month * 100 + stock_df['date'].dt.day 
#print(stock_df['float date']) # good

# Merging the dataframes (overall stocks)
merged_df1 = pd.merge(stock_df, subset_df_sorted, left_on = 'symbol', right_on = 'Ticker Symbol', how = 'inner') # good

print("new dataframe columns: \n" , merged_df1.columns.tolist())
print("new dataframe:\n", merged_df1) # good

dropped_columns = ['symbol', 'Ticker Symbol', 'date', 'float date', 'Period Ending', 'float Period Ending']

# Converting all the columns to numeric values
print("Before applying the function:", merged_df1.shape)
merged_df1 = convert_to_numeric_except(merged_df1, dropped_columns)
print("After applying the function:", merged_df1.shape) # good

"""
for symbol, group_df in merged_df1:
    drop_columns = ['symbol', 'date', 'Ticker Symbol', 'Period Ending']
    numeric_columns = [col for col in group_df.columns if col not in drop_columns]
    
    convert_to_numeric_allcols(group_df[numeric_columns])
"""


print("Data types of each column:") # good
for column in merged_df1.columns:
    print(f"{column}: {merged_df1[column].dtype}")


# CORRELATION MATRICES
# Use merged_df1 from here on out unless stated otherwise
# Create an overall correlation matrix to see which variables correlate with that stock the most
dropped_df = merged_df1.drop(columns = dropped_columns)
print("dropped df:\n", dropped_df) # good
print("dropped df columns:\n", dropped_df.columns.tolist()) # good

overall_correlation_matrix = dropped_df.corr()
print(overall_correlation_matrix) # good
#overall_correlation_matrix.to_excel('overall correlation matrix.xlsx', index = True) done

# Correlation matrix for each individual stock we are analyzing, which includes:
# -	Apple (APPL)
# -	Yum! Brands (YUM)
# -	Public Service Enterprise Group (PEG)
# -	Visa (V)
# -	Tractor Supply Co (TSCO)

chosen_stocks = ['YUM', 'PEG', 'TSCO', 'ABC']

filtered_data = merged_df1[merged_df1['symbol'].isin(chosen_stocks)] # filters by chosen stocks
filtered_data['symbol'] = filtered_data['symbol'].str.strip()
grouped_data = filtered_data.groupby('symbol')
print(grouped_data)

correlation_matrices = {}

for i, data in grouped_data:
    
    numeric_data = data.select_dtypes(include=np.number)

    corr_matrix = numeric_data.corr()
    correlation_matrices[i] = corr_matrix
"""
for stock, matrix in correlation_matrices.items():
    rounded_matrix = matrix.round(3)  # Round to 3 decimal points
    correlation_matrices[stock] = rounded_matrix
"""
    
print("Cencora (ABC) Correlation Matrix:\n", correlation_matrices['ABC'])
print("YUM Correlation Matrix: \n", correlation_matrices['YUM'])
print("PEG Correlation matrix:\n", correlation_matrices['PEG'])
print("TSCO Correlation matrix:\n", correlation_matrices['TSCO']) # has NaN w.r.t gross margin drop it from here

""" good dont regen they arent named w/ a 2 this would be a new file
correlation_matrices['ABC'].to_excel('ABC correlation matrix2.xlsx', index = True)
correlation_matrices['YUM'].to_excel('YUM correlation matrix2.xlsx', index = True)
correlation_matrices['PEG'].to_excel('PEG correlation matrix2.xlsx', index = True)
correlation_matrices['V'].to_excel('V correlation matrix2.xlsx', index = True)
correlation_matrices['TSCO'].to_excel('TSCO correlation matrix2.xlsx', index = True)
"""


"""
for symbol, group_df in stock_names: # good but doesnt really tell me anything
    drop_columns = ['symbol', 'date', 'float date']
    numeric_columns = [col for col in group_df.columns if col not in drop_columns]
    
    # Convert columns to numeric for the current stock DataFrame
    convert_to_numeric_allcols(group_df[numeric_columns])
    
    # Drop non-numeric columns and compute correlation matrix
    dropped_df = group_df.drop(columns=drop_columns)
    corr_matrix = dropped_df.corr()
    
    # Store the correlation matrix in a dictionary
    correlation_matrices[symbol] = corr_matrix
    
    for symbol, corr_matrix in correlation_matrices.items():
        print(f"Correlation matrix for stock: {symbol}")
        print(corr_matrix)
        print("\n")
"""
# Training and testing neueral net
# For overall data set
# Using dropped_df which is merged_df1 but without the nonnumerical columns
analyzed_columns = ['open', 'low', 'high', 'volume', 'Profit Margin', 'Gross Margin', 'Operating Margin', 'Total Revenue']
target_column = 'close'

X = dropped_df[analyzed_columns]
y = dropped_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_selected = X_train[analyzed_columns]
X_test_selected = X_test[analyzed_columns]

# Fitting data
overall_model = LinearRegression()
overall_model.fit(X_train_selected, y_train)

# Predicting model fit
overall_prediction = overall_model.predict(X_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y_test, overall_prediction)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")

# Displaying equation line
slope = overall_model.coef_[0]
intercept = overall_model.intercept_

# Create equation string for the regression line
equation = f"Regression Line: y = {slope:.2f}x + {intercept:.2f}"

# Add the equation text to the plot
plt.text(min(y_test), max(overall_prediction), equation, ha='left')

plt.legend()
plt.show()

# For each value 
# When open influences closing price the most
col1 = 'open'

# Extract the features and target variable
X1 = dropped_df[[col1]]
y1 = dropped_df[target_column]

# Split the data into training and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X1_train_selected = X1_train[[col1]].values.reshape(-1, 1)
X1_test_selected = X1_test[[col1]].values.reshape(-1, 1)

# Fitting data
model1 = LinearRegression()
model1.fit(X1_train_selected, y1_train)

# Predicting model fit
prediction1 = model1.predict(X1_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y1_test, prediction1, color = 'yellow')
plt.plot([min(y1_test), max(y1_test)], [min(y1_test), max(y1_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Open Price vs Close Price")

# Displaying equation line
slope1 = model1.coef_[0]
intercept1 = model1.intercept_

# Create equation string for the regression line
equation1 = f"Regression Line: y = {slope1:.2f}x + {intercept1:.2f}"

# Add the equation text to the plot
plt.text(min(y1_test), max(prediction1), equation1, ha='left')


plt.legend()
plt.show()

# When low influences closing price the most
col2 = 'low'

# Extract the features and target variable
X2 = dropped_df[[col2]]
y2 = dropped_df[target_column]

# Split the data into training and test sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X2_train_selected = X2_train[[col2]].values.reshape(-1, 1)
X2_test_selected = X2_test[[col2]].values.reshape(-1, 1)

# Fitting data
model2 = LinearRegression()
model2.fit(X2_train_selected, y2_train)

# Predicting model fit
prediction2 = model2.predict(X2_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y2_test, prediction2, color = 'orange')
plt.plot([min(y2_test), max(y2_test)], [min(y2_test), max(y2_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Low Price vs. Close Price")

# Displaying equation line
slope2 = model2.coef_[0]
intercept2 = model2.intercept_

# Create equation string for the regression line
equation2 = f"Regression Line: y = {slope2:.2f}x + {intercept2:.2f}"

# Add the equation text to the plot
plt.text(min(y2_test), max(prediction2), equation2, ha='left')

plt.legend()
plt.show()

# When high influences closing price the most
col3 = 'high'

# Extract the features and target variable
X3 = dropped_df[[col3]]
y3 = dropped_df[target_column]

# Split the data into training and test sets
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X3_train_selected = X3_train[[col3]].values.reshape(-1, 1)
X3_test_selected = X3_test[[col3]].values.reshape(-1, 1)

# Fitting data
model3 = LinearRegression()
model3.fit(X3_train_selected, y3_train)

# Predicting model fit
prediction3 = model3.predict(X3_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y3_test, prediction3, color = 'pink')
plt.plot([min(y3_test), max(y3_test)], [min(y3_test), max(y3_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values") # independent variable
plt.ylabel("Predicted Values") # close
plt.title("High Price vs. Close Price")

# Displaying equation line
slope3 = model3.coef_[0]
intercept3 = model3.intercept_

# Create equation string for the regression line
equation3 = f"Regression Line: y = {slope3:.2f}x + {intercept3:.2f}"

# Add the equation text to the plot
plt.text(min(y3_test), max(prediction3), equation3, ha='left')

plt.legend()
plt.show()


# When volume influences closing price the most
col4 = 'volume'

# Extract the features and target variable
X4 = dropped_df[[col4]]
y4 = dropped_df[target_column]

# Split the data into training and test sets
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X4_train_selected = X4_train[[col4]].values.reshape(-1, 1)
X4_test_selected = X4_test[[col4]].values.reshape(-1, 1)

# Fitting data
model4 = LinearRegression()
model4.fit(X4_train_selected, y4_train)

# Predicting model fit
prediction4 = model4.predict(X4_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y4_test, prediction4, color = 'blue')
plt.plot([min(y4_test), max(y4_test)], [min(y4_test), max(y4_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Volume vs. Close Price")

# Displaying equation line
slope4= model4.coef_[0]
intercept4 = model4.intercept_

# Create equation string for the regression line
equation4 = f"Regression Line: y = {slope4:.2f}x + {intercept4:.2f}"

# Add the equation text to the plot
plt.text(min(y4_test), max(prediction4), equation4, ha='left')

plt.legend()
plt.show()

# When profit margin influences closing price the most
col5 = 'Profit Margin'

# Extract the features and target variable
X5 = dropped_df[[col5]]
y5 = dropped_df[target_column]

# Split the data into training and test sets
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X5_train_selected = X5_train[[col5]].values.reshape(-1, 1)
X5_test_selected = X5_test[[col5]].values.reshape(-1, 1)

# Fitting data
model5 = LinearRegression()
model5.fit(X5_train_selected, y5_train)

# Predicting model fit
prediction5 = model5.predict(X5_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y5_test, prediction5, color = 'blue')
plt.plot([min(y5_test), max(y5_test)], [min(y5_test), max(y5_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Profit Margin vs. Close Price")

# Displaying equation line
slope5 = model5.coef_[0]
intercept5 = model5.intercept_

# Create equation string for the regression line
equation5 = f"Regression Line: y = {slope5:.2f}x + {intercept5:.2f}"

# Add the equation text to the plot
plt.text(min(y5_test), max(prediction5), equation5, ha='left')

plt.legend()
plt.show()

# When gross margin influences closing price the most
col6 = 'Gross Margin'

# Extract the features and target variable
X6 = dropped_df[[col6]]
y6 = dropped_df[target_column]

# Split the data into training and test sets
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X6_train_selected = X6_train[[col6]].values.reshape(-1, 1)
X6_test_selected = X6_test[[col6]].values.reshape(-1, 1)

# Fitting data
model6 = LinearRegression()
model6.fit(X6_train_selected, y6_train)

# Predicting model fit
prediction6 = model6.predict(X6_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y6_test, prediction6, color='blue')
plt.plot([min(y6_test), max(y6_test)], [min(y6_test), max(y6_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Gross Margin vs. Close Price")

# Displaying equation line
slope6 = model6.coef_[0]
intercept6 = model6.intercept_

# Create equation string for the regression line
equation6 = f"Regression Line: y = {slope6:.2f}x + {intercept6:.2f}"

# Add the equation text to the plot
plt.text(min(y6_test), max(prediction6), equation6, ha='left')

plt.legend()
plt.show()

# When Operating margin influences closing price the most
col7 = 'Operating Margin'

# Extract the features and target variable
X7 = dropped_df[[col7]]
y7 = dropped_df[target_column]

# Split the data into training and test sets
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X7_train_selected = X7_train[[col7]].values.reshape(-1, 1)
X7_test_selected = X7_test[[col7]].values.reshape(-1, 1)

# Fitting data
model7 = LinearRegression()
model7.fit(X7_train_selected, y7_train)

# Predicting model fit
prediction7 = model7.predict(X7_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y7_test, prediction7, color='blue')
plt.plot([min(y7_test), max(y7_test)], [min(y7_test), max(y7_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Operating Margin vs. Close Price")

# Displaying equation line
slope7 = model7.coef_[0]
intercept7 = model7.intercept_

# Create equation string for the regression line
equation7 = f"Regression Line: y = {slope7:.2f}x + {intercept7:.2f}"

# Add the equation text to the plot
plt.text(min(y7_test), max(prediction7), equation7, ha='left')

plt.legend()
plt.show()

# When total revenue influences closing price the most
col8 = 'Total Revenue'

# Extract the features and target variable
X8 = dropped_df[[col8]]
y8 = dropped_df[target_column]

# Split the data into training and test sets
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=0.2, random_state=42)

# Reshape the arrays to be 2D
X8_train_selected = X8_train[[col8]].values.reshape(-1, 1)
X8_test_selected = X8_test[[col8]].values.reshape(-1, 1)

# Fitting data
model8 = LinearRegression()
model8.fit(X8_train_selected, y8_train)

# Predicting model fit
prediction8 = model8.predict(X8_test_selected)

# Plotting actual vs. predicted values
plt.scatter(y8_test, prediction8, color='blue')
plt.plot([min(y8_test), max(y8_test)], [min(y8_test), max(y8_test)], color='black', linewidth=1.75)  # Regression line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Total Revenue vs. Close Price")

# Displaying equation line
slope8 = model8.coef_[0]
intercept8 = model8.intercept_

# Create equation string for the regression line
equation8 = f"Regression Line: y = {slope8:.2f}x + {intercept8:.2f}"

# Add the equation text to the plot
plt.text(min(y8_test), max(prediction8), equation8, ha='left')

plt.legend()
plt.show()

# For each correlation matrix
 

# Fancy stock graphs
# On overall data set, too big
"""
dropped_columns2 = ['symbol', 'Ticker Symbol', 'float date', 'Period Ending', 'float Period Ending']
includingdate_df = merged_df1.drop(columns=dropped_columns2)
includingdate_df['date'] = pd.to_datetime(includingdate_df['date'])  # Convert to datetime if not already in datetime format
includingdate_df.set_index('date', inplace=True)  # Set 'date' column as index
apdict = mpf.make_addplot(overall_prediction, color='orange')
mpf.plot(includingdate_df, type='candle', addplot=apdict, title='Financial Data with Linear Regression')
plt.show()
"""

# For chosen stocks, YUM, TSCO, PEG, ABC
dropped_columns2 = ['symbol', 'Ticker Symbol', 'float date', 'Period Ending', 'float Period Ending']
includingdate_df = merged_df1.drop(columns=dropped_columns2)
includingdate_df['date'] = pd.to_datetime(includingdate_df['date'])  # Convert to datetime if not already in datetime format
includingdate_df.set_index('date', inplace=True)  # Set 'date' column as index

"""
for symbol, data in grouped_data:
    data = data.set_index('date')  # Assuming 'date' column contains datetime values
    
    # Prepare data for linear regression
    X = data.index.factorize()[0].reshape(-1, 1)  # Using index as the x-axis
    y = data['close']
       
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    regression_line = model.predict(X)
       
    # Prepare linear regression line for plotting
    data['regression_line'] = regression_line
       
    # Plot candlestick chart with linear regression line
    apdict = mpf.make_addplot(data['regression_line'], color='red', width=1)
    mpf.plot(data, type='candle', addplot=apdict, title=f'Financial Data with Linear Regression - {symbol}')
    apdict = mpf.make_addplot(data['close'], color='orange')
    mpf.plot(data, type='candle', addplot=apdict, title=f'Financial Data with Linear Regression - {symbol}')
    plt.show()
    """
for symbol, data in grouped_data: # GOOD
    data = data.set_index('date')  # Assuming 'date' column contains datetime values
    
    # Prepare data for linear regression
    X = data.index.factorize()[0].reshape(-1, 1)  # Using index as the x-axis
    y = data['close']
       
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    regression_line = model.predict(X)
    
    
    # Prepare linear regression line for plotting
    data['regression_line'] = regression_line
       
    # Get coefficients (slope and intercept)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Print the equation of the linear regression line
    equation = f"Regression Line: y = {slope:.2f}x + {intercept:.2f}"
    print(f"For symbol {symbol}:")
    print(equation)
    
    # Plot candlestick chart with linear regression line
    apdicts = [
        mpf.make_addplot(data['close'], color='orange'),
        mpf.make_addplot(data['regression_line'], color='black', width=1)
    ]
    mpf.plot(data, type='candle', addplot=apdicts, title=f'Linear Regression and Stock Data - {symbol}')
    plt.show()
 
# Zooming in on specific dates for closer look at what the graph looks like
start_date = '2014-01-01'
end_date = '2014-01-31' # had it as jan12015 first

for symbol, data in grouped_data:
    data = data.set_index('date')  # Assuming 'date' column contains datetime values
    
    # Slice the DataFrame based on the date range
    zoomed_data = data.loc[start_date:end_date]
    
    # Plot zoomed-in candlestick chart for each stock
    mpf.plot(zoomed_data, type='candle', title=f'Zoomed-in View - {symbol}')
    plt.show()
    
"""
for stock, data in grouped_data:
    # Create a plot for each stock
    apdict = mpf.make_addplot(data['close'], color='orange')  # Replace 'close' with your target column
    mpf.plot(data, type='candle', addplot=apdict, title=f"{stock} Stock Data")
"""
"""
# Plotting residuals (difference between actual and predicted values)
residuals1 = y_test - prediction1
plt.scatter(prediction1, residuals1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals Plot (Linear Regression)")
plt.axhline(y=0, color='r', linestyle='-')  # Adding a horizontal line at y=0
plt.show()
"""

""" #good
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
"""


# For each stock I analyzed

"""
# PCA to handle multicolinearity
dropped_dfcol = dropped_df.columns
scaler = StandardScaler()
dropped_df[dropped_dfcol] = scaler.fit_transform(dropped_df[dropped_dfcol])

pca1 = PCA(n_components = 2)
principal_components1 = pca1.fit_transform(dropped_df)
explained_variance1 = pca1.explained_variance_ratio_ # Shows amount of variance for each component

print(pca1)
print(principal_components1)

# Plotting PCA (biplot)
for i, component in enumerate(pca1.components_):
    plt.arrow(0,0, component[0], component[1], head_width = 0.2, head_length = 0.1, fc = '#57C5EC', ec = '#57C5EC')
    plt.text(component[0], component[1], f"Eigenvector {i+1}" , fontsize = 12)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Biplot for PCA (2D)')
plt.grid(True) 
plt.show()
"""
# Graph for PCA (dont do)


# Linear Regression
# We will be looking at each individual parameter and modeling the linear regression
#



