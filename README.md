# Airline Passenger Traffic Forecasting

This repository contains a project focused on forecasting airline passenger traffic using various time series forecasting methods. The dataset used in this project is the Airline Passenger Traffic dataset.

Table of Contents

Data Preprocessing
Mean Imputation
Linear Interpolation
Outlier Detection
Box Plot and Interquartile Range
Histogram Plot
Train-Test Split
Time Series Forecasting Models
Naive Method
Simple Average Method
Simple Moving Average Method
Exponential Smoothing Methods
Simple Exponential Smoothing
Holt's Method with Trend
Holt-Winters Additive and Multiplicative Methods with Trend and Seasonality
Data Preprocessing

Imputing Missing Values
To handle missing values in the dataset, we employed the following techniques:

Mean Imputation: Replaced missing values with the mean of the non-missing values in the dataset.
Linear Interpolation: Used linear interpolation to estimate missing values based on the values before and after the missing entries.
Outlier Detection

Outliers were identified and handled using the following methods:

Box Plot and Interquartile Range (IQR): Used box plots to visualize data distribution and identify outliers. Outliers were determined as data points lying beyond 1.5 times the IQR from the first and third quartiles.
Histogram Plot: Utilized histograms to understand the frequency distribution of the data and identify potential outliers visually.
Train-Test Split

The dataset was divided into training and testing sets to evaluate the performance of the forecasting models. The training set was used to build the models, and the testing set was used to assess their accuracy.

Time Series Forecasting Models

Several basic time series forecasting models were implemented, including:

Naive Method: Assumes that the next observation will be the same as the last observation.
Simple Average Method: Uses the average of all past observations as the forecast.
Simple Moving Average Method: Calculates the average of a fixed number of past observations to predict the next value.
Exponential Smoothing Methods

Advanced exponential smoothing methods were used for more accurate forecasting:

Simple Exponential Smoothing: Applies exponential weights to past observations, giving more weight to recent observations.
Holt's Method with Trend: Extends simple exponential smoothing to capture linear trends in the data.
Holt-Winters Additive and Multiplicative Methods with Trend and Seasonality: Further extends Holt's method to account for both trend and seasonality, with additive and multiplicative options depending on the nature of the seasonality.
