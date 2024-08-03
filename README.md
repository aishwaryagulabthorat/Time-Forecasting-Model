# Airline Passenger Traffic Forecasting

This repository contains a project focused on forecasting airline passenger traffic using various time series forecasting methods. The dataset used in this project is the Airline Passenger Traffic dataset.

## Step Performed -

### 1. Data Preprocessing
Mean Imputation - Replaced missing values with the mean of the non-missing values in the dataset.
Linear Interpolation - Used linear interpolation to estimate missing values based on the values before and after the missing entries.
Outlier Detection

### 2. Outlier Detection
Box Plot and Interquartile Range - Used box plots to visualize data distribution and identify outliers. Outliers were determined as data points lying beyond 1.5 times the IQR from the first and third quartiles.
Histogram Plot - Utilized histograms to understand the frequency distribution of the data and identify potential outliers visually.
Train-Test Split

### 3. Train-Test Split

### 4. Time Series Forecasting Models
Naive Method - Assumes that the next observation will be the same as the last observation.
Simple Average Method - Uses the average of all past observations as the forecast.
Simple Moving Average Method - Calculates the average of a fixed number of past observations to predict the next value.
Exponential Smoothing Methods

### 5. Exponential Smoothing Methods
Simple Exponential Smoothing - Applies exponential weights to past observations, giving more weight to recent observations.
Holt's Method with Trend - Extends simple exponential smoothing to capture linear trends in the data.
Holt-Winters Additive and Multiplicative Methods with Trend and Seasonality - Further extends Holt's method to account for both trend and seasonality, with additive and multiplicative options depending on the nature of the seasonality.


### Results

The results of the various forecasting methods are compared based on their accuracy metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE). The models' performance highlights the effectiveness of different approaches in capturing the underlying patterns of the airline passenger traffic data.
