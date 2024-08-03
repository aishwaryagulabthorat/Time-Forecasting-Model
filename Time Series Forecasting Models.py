#!/usr/bin/env python
# coding: utf-8

# # Time series forecasting

# ### Considered Airline Passenger Traffic Dataset
# 
# 
# ### In Data Preprocessing to impute missing values -
# ### 1. Mean Imputation
# ### 2. Linear interpolation
# 
# 
# 
# ### For outlier Detection -
# ### 1. Box plot and interquartile range
# ### 2. Histogram plot
# 
# 
# 
# ### Train - Test Split
# 
# 
# 
# 
# ### Time Series Forecasting Models -
# 
# ### 1. Naive Method
# ### 2. Simple Average Method
# ### 3. Simple moving average method
# 
# 
# 
# 
# ### Exponential smoothing methods
# 
# ### 1. Simple exponential smoothing
# ### 2. Holt's method with trend
# ### 3. Holt Winter's additive and multiplicative method with trend and seasonality

# ## Import required packages

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# ## Import time series data: Airline passenger traffic

# In[2]:


data = pd.read_csv('/Users/aishwaryathorat/Movies/MS Courses/Upg/Time Series Forecasting/Python/airline-passenger-traffic.csv', header = None)
data.columns = ['Month','Passengers']
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data = data.set_index('Month')
data.head(12)


# # Time series analysis

# ## Plot time series data

# In[3]:


data.plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Airline passenger traffic')
plt.show(block=False)


# ## Missing value treatment

# Going to try two missing value imputation techniques and select the most appropriate for our data.

# ### Mean imputation

# In[4]:


data = data.assign(Passengers_Mean_Imputation=data.Passengers.fillna(data.Passengers.mean()))
data[['Passengers_Mean_Imputation']].plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Airline passenger traffic: Mean imputation')
plt.show(block=False)


# ### Linear interpolation

# In[5]:


data = data.assign(Passengers_Linear_Interpolation=data.Passengers.interpolate(method='linear'))
data[['Passengers_Linear_Interpolation']].plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Airline passenger traffic: Linear interpolation')
plt.show(block=False)


# ### Use linear interpolation to impute missing values

# For our data we can clearly see from above graphs that linear interpolation works better for our data because mean imputation is filling null values with slighly extreme values.

# In[6]:


data['Passengers'] = data['Passengers_Linear_Interpolation']
data.drop(columns=['Passengers_Mean_Imputation','Passengers_Linear_Interpolation'],inplace=True)


# ## Outlier detection

# ### Box plot and interquartile range

# In[7]:


import seaborn as sns
fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=data['Passengers'],whis=1.5)


# ### Histogram plot

# In[8]:


fig = data.Passengers.hist(figsize = (12,4))


# ## Time series Decomposition

# ### Additive seasonal decomposition

# In[9]:


from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 12, 8
decomposition = sm.tsa.seasonal_decompose(data.Passengers, model='additive') # additive seasonal index
fig = decomposition.plot()
plt.show()


# ### Multiplicative seasonal decomposition

# In[10]:


decomposition = sm.tsa.seasonal_decompose(data.Passengers, model='multiplicative') # multiplicative seasonal index
fig = decomposition.plot()
plt.show()


# # Build and evaluate time series forecast

# ## Split time series data into training and test set

# In[11]:


train_len = 120
train = data[0:train_len] # first 120 months as training set
test = data[train_len:] # last 24 months as out-of-time test set


# # Simple time series methods

# In[12]:


data


# ## Naive method

# In[13]:


y_hat_naive = test.copy()
y_hat_naive['naive_forecast'] = train['Passengers'][train_len-1]


# In[14]:


y_hat_naive


# In[15]:


train


# ### Plot train, test and forecast

# In[16]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')
plt.legend(loc='best')
plt.title('Naive Method')
plt.show()


# So, here we can clearly see the green line got extrapolated from 1958 to 1960. and we ended up underpredicting because we can clearly see numbers are high in the test data (orange highlighted).

# ### Calculate RMSE and MAPE

# In[17]:


from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_naive['naive_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_naive['naive_forecast'])/test['Passengers'])*100,2)

results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
results = results[['Method', 'RMSE', 'MAPE']]
results


# ## Simple average method

# In[18]:


y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Passengers'].mean()


# In[ ]:





# ### Plot train, test and forecast

# In[19]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Simple average forecast')
plt.legend(loc='best')
plt.title('Simple Average Method')
plt.show()


# so here also we are underpredicting. we are simply taking the average of the train data and predicting values.  this data has seasonality and simple average method is not exactly capturing it. we are underpredicting the value. 

# ### Calculate RMSE and MAPE

# In[20]:


rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_avg['avg_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_avg['avg_forecast'])/test['Passengers'])*100,2)

tempResults = pd.DataFrame({'Method':['Simple average method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ## Simple moving average method

# In[21]:


y_hat_sma = data.copy()
ma_window = 12
y_hat_sma['sma_forecast'] = data['Passengers'].rolling(ma_window).mean()

y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1] 
#our training set ends in 1958, and data of 1959 and 1960 is test data. and we cannot really predict values 
# of 1959 and 1969 using data fro training dataset. so for test dataset forecast will remain same as the 
# forescat of the last month of the train dataset


# In[22]:


y_hat_sma


# ### Plot train, test and forecast

# In[23]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()


# ### Calculate RMSE and MAPE

# In[24]:


rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_sma['sma_forecast'][train_len:])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_sma['sma_forecast'][train_len:])/test['Passengers'])*100,2)

tempResults = pd.DataFrame({'Method':['Simple moving average forecast'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# what if we take 6 month window

# In[25]:


y_hat_sma = data.copy()
ma_window = 6
y_hat_sma['sma_forecast_6'] = data['Passengers'].rolling(ma_window).mean()

y_hat_sma['sma_forecast_6'][train_len:] = y_hat_sma['sma_forecast_6'][train_len-1] 
#our training set ends in 1958, and data of 1959 and 1960 is test data. and we cannot really predict values 
# of 1959 and 1969 using data fro training dataset. so for test dataset forecast will remain same as the 
# forescat of the last month of the train dataset


# In[26]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_sma['sma_forecast_6'], label='Simple moving average forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
plt.show()


# we can clearly see with the smaller window it is following the seasonal pattern smaller the window size higher the trend but more noise will be there. but the in test data values are higher than the actual values.

# # Exponential smoothing methods

# ## Simple exponential smoothing

# In[27]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(train['Passengers'])

model_fit = model.fit(smoothing_level=0.2,optimized=False) #as we are manually mentioning smoothing level 0.2 we 
# will keep optimized False because otherwise if we keep true it will calculate optimal smoothing level and 
# apply that value

#smoothing level 0.2 is the weight assigned to the last avialable data and all the previous observations will get
#(1-0.2) weight

model_fit.params
y_hat_ses = test.copy()
y_hat_ses['ses_forecast'] = model_fit.forecast(24) #forecast for next 24 months


# ### Plot train, test and forecast

# In[28]:


plt.figure(figsize=(12,4))
plt.plot(train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_ses['ses_forecast'], label='Simple exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Simple Exponential Smoothing Method')
plt.show()

#in the graph we can see we are predicting the level and not pattern. and out levelled value is more or 
# less matching the test dataset values


# ### Calculate RMSE and MAPE

# In[29]:


rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_ses['ses_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_ses['ses_forecast'])/test['Passengers'])*100,2)

tempResults = pd.DataFrame({'Method':['Simple exponential smoothing forecast'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results


# ## Holt's method with trend

# In[30]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(np.asarray(train['Passengers']) ,seasonal_periods=12 ,trend='additive', seasonal=None) #seasonal pattern is every 12 months
#trend is linear trend that is why we used trend = additive, linear as in gradually increasing in one direction

model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.01, optimized=False)
print(model_fit.params)
y_hat_holt = test.copy()
y_hat_holt['holt_forecast'] = model_fit.forecast(len(test)) #len(test) is the total test period so 24 months or whiever data is present in test set


# ### Plot train, test and forecast

# In[31]:


plt.figure(figsize=(12,4))
plt.plot( train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s exponential smoothing forecast')
plt.legend(loc='best')
plt.title('Holt\'s Exponential Smoothing Method')
plt.show()


# here the forecast is a straight line but it is sloping upwards. because this modethod captures both level and trend. we can see thta the level is pretty much accurate and we are also getting the upward trending line that shows we are also capturing the trend of the data.

# ### Calculate RSME and MAPE

# In[32]:


rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_holt['holt_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_holt['holt_forecast'])/test['Passengers'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt\'s exponential smoothing method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# changed the smoothing slope from 0.01 to 0.1 and saw the results got a very straight line, which shows we gave very high weight to the recent observations. and because we assigned high weight to recent observationas it cancells out the trend of the previous observations. and gave us the straight line.

# ## Holt Winters' additive method with trend and seasonality

# In[33]:


y_hat_hwa = test.copy()
model = ExponentialSmoothing(np.asarray(train['Passengers']) ,seasonal_periods=12 ,trend='add', seasonal='add')
model_fit = model.fit(optimized=True) # find the value of alpha beta gama which reduces the sum of squared errors
print(model_fit.params)
y_hat_hwa['hw_forecast'] = model_fit.forecast(24)


# ### Plot train, test and forecast

# In[34]:


plt.figure(figsize=(12,4))
plt.plot( train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s additive forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Additive Method')
plt.show()


# ### Calculate RMSE and MAPE

# In[35]:


rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_hwa['hw_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_hwa['hw_forecast'])/test['Passengers'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt Winters\' additive method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results


# ## Holt Winter's multiplicative method with trend and seasonality

# In[36]:


y_hat_hwm = test.copy()
model = ExponentialSmoothing(np.asarray(train['Passengers']) ,seasonal_periods=12 ,trend='add', seasonal='mul')
model_fit = model.fit(optimized=True)
print(model_fit.params)
y_hat_hwm['hw_forecast'] = model_fit.forecast(24)


# ### Plot train, test and forecast

# In[37]:


plt.figure(figsize=(12,4))
plt.plot( train['Passengers'], label='Train')
plt.plot(test['Passengers'], label='Test')
plt.plot(y_hat_hwm['hw_forecast'], label='Holt Winters\'s mulitplicative forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Mulitplicative Method')
plt.show()


# ### Calculate RMSE and MAPE

# In[38]:


rmse = np.sqrt(mean_squared_error(test['Passengers'], y_hat_hwm['hw_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Passengers']-y_hat_hwm['hw_forecast'])/test['Passengers'])*100,2)

tempResults = pd.DataFrame({'Method':['Holt Winters\' multiplicative method'], 'RMSE': [rmse],'MAPE': [mape] })
results = pd.concat([results, tempResults])
results = results[['Method', 'RMSE', 'MAPE']]
results

