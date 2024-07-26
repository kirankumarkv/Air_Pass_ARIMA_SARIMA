# -*- coding: utf-8 -*-
"""ARIMA_SARIMA_TEST.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RFXkox7MDnluu6BDEQwExIqazmcfZCtP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)

data.head()

data.tail()

#Check for Stationarity
#TO apply ARIMA, the time series should be stationary
from statsmodels.tsa.stattools import adfuller

#Perform ADF test
result= adfuller(data['Passengers'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'{key}: {value}')

# If p-value > 0.05, the series is non-stationary
if result[1] > 0.05:
    print('Series is non-stationary')
else:
    print('Series is stationary')

data_series = data['Passengers']  # Replace 'Passengers' with your actual column name

# Differencing the data
data_diff = data_series.diff().dropna()

# Perform ADF test on differenced data
result_diff = adfuller(data_diff)

# Print test statistic and p-value
print('ADF Statistic (differenced):', result_diff[0])
print('p-value (differenced):', result_diff[1])

# Plot the differenced data
plt.plot(data_diff)
plt.title('Differenced Data')
plt.xlabel('Date')
plt.ylabel('Differenced Passengers')
plt.show()

# Interpretation
if result_diff[0] < result_diff[4]['5%'] and result_diff[1] < 0.05:
    print('The data is stationary after differencing.')
else:
    print('The data is not stationary after differencing.')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Determine the order of the ARIMA model using ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(data_diff, ax=axes[0])
plot_pacf(data_diff, ax=axes[1])
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Example data generation (Replace this with your actual data)
np.random.seed(0)
date_rng = pd.date_range(start='1/1/2020', end='1/1/2024', freq='M')
df = pd.DataFrame(date_rng, columns=['date'])
df['value'] = np.random.randn(len(df))
df.set_index('date', inplace=True)

# 1. Data Preprocessing: Check for stationarity and make stationary if needed
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')

test_stationarity(df['value'])

# Differencing if necessary
df['value_diff'] = df['value'].diff().dropna()
test_stationarity(df['value_diff'].dropna())

# 2. Model Selection: ACF and PACF plots
plot_acf(df['value_diff'].dropna())
plot_pacf(df['value_diff'].dropna())
plt.show()

# 3. Model Training and Validation: Split data
train_size = int(len(df) * 0.8)
train, test = df['value'][:train_size], df['value'][train_size:]

# 4. Model Diagnostics: Fit ARIMA model and check residuals
model = ARIMA(train, order=(1,1,1))
fitted_model = model.fit()
print(fitted_model.summary())

# Residuals analysis
residuals = fitted_model.resid
plot_acf(residuals)
plot_pacf(residuals)
plt.show()

# 5. Parameter Tuning: Grid search for best p, d, q
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
best_aic = np.inf
best_params = None
for param in pdq:
    try:
        model = ARIMA(train, order=param)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = param
    except:
        continue
print('Best ARIMA parameters:', best_params)

# 6. Model Evaluation: Evaluate on validation set
best_model = ARIMA(train, order=best_params).fit()
forecast = best_model.forecast(steps=len(test))
plt.plot(test.index, test, label='True')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.show()

# Metrics
rmse = np.sqrt(np.mean((forecast - test)**2))
print('RMSE:', rmse)

# 7. Incorporate Exogenous Variables (Example with one exogenous variable)
# Assuming df has an 'exog' column with external variable
if 'exog' in df.columns:
    model = ARIMA(train, order=best_params, exog=train['exog'])
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test), exog=test['exog'])
    plt.plot(test.index, test, label='True')
    plt.plot(test.index, forecast, label='Forecast with Exog')
    plt.legend()
    plt.show()

# 8. Update Model Regularly: Example function to retrain model
def update_model(new_data):
    global best_model
    best_model = best_model.append(new_data)
    best_model = best_model.fit()
    return best_model

forecast = best_model.forecast(steps=len(test)) # Match the length of test data
plt.plot(forecast.index, forecast, label='Future Forecast')
plt.legend()
plt.show()

# 10. Documentation and Reporting
import seaborn as sns
sns.set()
plt.figure(figsize=(10, 6))
plt.plot(df['value'], label='Original')
plt.plot(test.index, forecast, label='Forecast')  # Use test.index to match the forecast index

plt.title('ARIMA Forecast')
plt.legend()
plt.show()

"""# **Check for Stationarity (ADF Test) and Plot**"""

from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Passengers'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'{key}: {value}')

if result[1] > 0.05:
    print('Series is non-stationary')
else:
    print('Series is stationary')

# Plot the original data
data['Passengers'].plot(title='Original Data', figsize=(12, 6))
plt.show()

"""# **Difference the Data and Plot**"""

data_series = data['Passengers']
data_diff = data_series.diff().dropna()

result_diff = adfuller(data_diff)
print(f'ADF Statistic (differenced): {result_diff[0]}')
print(f'p-value (differenced): {result_diff[1]}')

import matplotlib.pyplot as plt
plt.plot(data_diff)
plt.title('Differenced Data')
plt.xlabel('Date')
plt.ylabel('Differenced Passengers')
plt.show()

"""# **Determine the Order of the ARIMA Model (ACF and PACF Plots):**"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data_diff)
plot_pacf(data_diff)
plt.show()

"""# **Fit the ARIMA Model**"""

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data_series, order=(2, 1, 2))  # Replace (p, d, q) with determined values
model_fit = model.fit()
print(model_fit.summary())

"""# **Make Predictions and Plot:**




"""

forecast = model_fit.forecast(steps=10)  # Adjust steps as needed
print(forecast)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Log transformation to stabilize the variance
data_log = np.log(data['Passengers'])

# Fit ARIMA model
model = ARIMA(data_log, order=(2, 1, 2))
arima_model = model.fit()

# Summary of the model
print(arima_model.summary())

# Plotting the original series, fitted values, and forecast
plt.figure(figsize=(14, 8))

# Original time series
plt.plot(data_log, label='Original Series')

# Fitted values
plt.plot(arima_model.fittedvalues, color='red', label='Fitted Values')

# Forecast future values (e.g., next 24 months)
forecast_steps = 24
forecast = arima_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data_log.index[-1], periods=forecast_steps + 1, freq='MS')[1:]

# Combine original, fitted values, and forecast in the same plot
plt.plot(forecast_index, forecast, color='green', label='Forecast')

plt.legend()
plt.title('Original Series, Fitted Values, and Forecast')
plt.xlabel('Date')
plt.ylabel('Log(Passengers)')
plt.show()

# Create the ACF and PACF plots
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot ACF
plot_acf(data_log, ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF) - Log Transformed Data')

# Plot PACF
plot_pacf(data_log, ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF) - Log Transformed Data')

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Log transformation to stabilize the variance
data_log = np.log(data['Passengers'])

# Fit ARIMA model
model = ARIMA(data_log, order=(4, 1, 4))
arima_model = model.fit()

# Summary of the model
print(arima_model.summary())

# Plotting the original series, fitted values, and forecast
plt.figure(figsize=(14, 8))

# Original time series
plt.plot(data_log, label='Original Series')

# Fitted values
plt.plot(arima_model.fittedvalues, color='red', label='Fitted Values')

# Forecast future values (e.g., next 24 months)
forecast_steps = 24
forecast = arima_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data_log.index[-1], periods=forecast_steps + 1, freq='MS')[1:]

# Combine original, fitted values, and forecast in the same plot
plt.plot(forecast_index, forecast, color='green', label='Forecast')

plt.legend()
plt.title('Original Series, Fitted Values, and Forecast')
plt.xlabel('Date')
plt.ylabel('Log(Passengers)')
plt.show()

# Create the ACF and PACF plots
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot ACF
plot_acf(data_log, ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF) - Log Transformed Data')

# Plot PACF
plot_pacf(data_log, ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF) - Log Transformed Data')

plt.tight_layout()
plt.show()

"""# **Seasonal Decomposition**"""

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data_log, model='additive')
result.plot()
plt.show()

"""# **Grid Search for ARIMA Hyperparameters**"""

import itertools
import warnings
warnings.filterwarnings("ignore")

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

best_aic = float("inf")
best_order = None
best_mdl = None

for param in pdq:
    try:
        temp_mdl = ARIMA(data_log, order=param).fit()
        if temp_mdl.aic < best_aic:
            best_aic = temp_mdl.aic
            best_order = param
            best_mdl = temp_mdl
    except:
        continue

print(f'Best ARIMA order: {best_order}')
print(f'Best AIC: {best_aic}')

"""# **Residual Diagnostics**"""

residuals = arima_model.resid
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=10)
print('Ljung-Box test p-values:')
# Access the p-values using the column name instead of a numerical index
print(lb_test['lb_pvalue'])

"""# **Rolling Forecast Origin (Backtesting):**"""

train_size = int(len(data_log) * 0.8)
train, test = data_log[:train_size], data_log[train_size:]

history = [x for x in train]
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(2, 1, 2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test[t])

plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.legend()
plt.show()

"""# **Ensemble Methods**"""

from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(data_log, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)).fit()
print(sarima_model.summary())

"""# **Forecasting Intervals**"""

forecast_result = arima_model.get_forecast(steps=24, alpha=0.05)

# Extract the predicted mean, standard error, and confidence interval
forecast = forecast_result.predicted_mean
stderr = forecast_result.se_mean
conf_int = forecast_result.conf_int()

forecast_index = pd.date_range(start=data_log.index[-1], periods=24 + 1, freq='MS')[1:]

plt.figure(figsize=(12, 8))
plt.plot(data_log, label='Original Series')
plt.plot(arima_model.fittedvalues, color='red', label='Fitted Values')
plt.plot(forecast_index, forecast, color='green', label='Forecast')

# Access the confidence intervals correctly for a MultiIndex DataFrame
lower_conf = conf_int.iloc[:, 0]  # Access the first column (lower bound)
upper_conf = conf_int.iloc[:, 1]  # Access the second column (upper bound)

plt.fill_between(forecast_index, lower_conf, upper_conf, color='green', alpha=0.2)
plt.legend()
plt.show()

# Forecast future values (e.g., next 24 months)
forecast_steps = 24
forecast_log = arima_model.get_forecast(steps=forecast_steps)
forecast_values_log = forecast_log.predicted_mean
forecast_conf_int = forecast_log.conf_int()

# Convert log forecasts back to original scale
forecast_values = np.exp(forecast_values_log)
forecast_conf_int = np.exp(forecast_conf_int)

# Create forecast index
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='MS')[1:]

# Plotting the original series, fitted values, and forecast
plt.figure(figsize=(14, 8))

# Original time series
plt.plot(data['Passengers'], label='Original Series')

# Fitted values (back-transformed)
fitted_values_log = arima_model.fittedvalues
fitted_values = np.exp(fitted_values_log)
plt.plot(fitted_values, color='red', label='Fitted Values')

# Combine original, fitted values, and forecast in the same plot
plt.plot(forecast_index, forecast_values, color='green', label='Forecast')

# Plot confidence intervals
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.2)

plt.legend()
plt.title('Original Series, Fitted Values, and Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.show()

"""# **SARIMA**


*   Used for Forecasting current value and seasonal value(12th value)
*  P Q D S=12








"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the data
data_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(data_url, parse_dates=['Month'], index_col='Month')

# Seasonal differencing (for monthly data with annual seasonality, period=12)
df['Seasonal_Difference'] = df['Passengers'].diff(12)

# Dropping NA values after differencing
seasonal_diff = df['Seasonal_Difference'].dropna()

# Plot ACF and PACF for the seasonally differenced data
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# ACF plot
plot_acf(seasonal_diff, lags=48, ax=ax[0])
ax[0].set_title('ACF of Seasonally Differenced Data')

# PACF plot
plot_pacf(seasonal_diff, lags=48, ax=ax[1])
ax[1].set_title('PACF of Seasonally Differenced Data')

plt.tight_layout()
plt.show()

# Seasonal ARIMA model
# Define the model
sarima_model = sm.tsa.statespace.SARIMAX(df['Passengers'],
                                         order=(1, 1, 1),
                                         seasonal_order=(1, 1, 1, 12),
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)

# Fit the model
sarima_results = sarima_model.fit()

# Summary of the model
print(sarima_results.summary())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Assuming df is your DataFrame with a 'Passengers' column
# and an index of DateTime type.

# Log-transform the 'Passengers' data
df['Log_Passengers'] = np.log(df['Passengers'])

# Split data into training and testing sets
train_data = df['Log_Passengers'][:int(0.8*len(df))]
test_data = df['Log_Passengers'][int(0.8*len(df)):]

# Fit SARIMA model
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model = model.fit(disp=False)

# Predicting in-sample (fitted values) and out-of-sample (forecast)
fitted_values = sarima_model.fittedvalues
forecast_steps = len(test_data)
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = test_data.index
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Log_Passengers'], label='Original Series', color='blue')
plt.plot(train_data.index, fitted_values, label='Fitted Values', color='red')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.2)

plt.title('SARIMA Model - Original Series and Fitted Values')
plt.xlabel('Date')
plt.ylabel('Log of Number of Passengers')
plt.legend()
plt.show()

"""# **CATFISH CSV-ARIMA & SARIMA**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the data (assuming 'catfish.csv' is already loaded into a DataFrame 'data')
data = pd.read_csv('catfish.csv')

# Convert the Date column to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Log-transform the 'Total' data
data['Log_Total'] = np.log(data['Total'])

# Split data into training and testing sets
train_data = data['Log_Total'][:int(0.8*len(data))]
test_data = data['Log_Total'][int(0.8*len(data)):]

# Fit SARIMA model
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model = model.fit(disp=False)

# Predicting in-sample (fitted values) and out-of-sample (forecast)
fitted_values = sarima_model.fittedvalues
forecast_steps = len(test_data)
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = test_data.index
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Total'], label='Original Series', color='blue')
plt.plot(train_data.index, fitted_values, label='Fitted Values', color='red')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='lightgreen', alpha=0.3)

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load the dataset
data = pd.read_csv('catfish.csv', parse_dates=['Date'], index_col='Date')

# Replace 'Passengers' with your actual column name, e.g., 'Catfish Sales'
data_series = data['Total']

# Differencing the data
data_diff = data_series.diff().dropna()

# Perform ADF test on differenced data
result_diff = adfuller(data_diff)

# Print test statistic and p-value
print('ADF Statistic (differenced):', result_diff[0])
print('p-value (differenced):', result_diff[1])

# Plot the differenced data
plt.plot(data_diff)
plt.title('Differenced Data')
plt.xlabel('Date')
plt.ylabel('Differenced Total')
plt.show()

# Interpretation
if result_diff[0] < result_diff[4]['5%'] and result_diff[1] < 0.05:
    print('The data is stationary after differencing.')
else:
    print('The data is not stationary after differencing.')

# Determine the order of the ARIMA model using ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(data_diff, ax=axes[0])
plot_pacf(data_diff, ax=axes[1])
plt.show()

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data_series, order=(2, 1, 2))  # Replace (p, d, q) with determined values
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=10)  # Adjust steps as needed
print(forecast)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA



# Log transformation to stabilize the variance
data_log = np.log(data['Total'])

# Check for stationarity with ADF test
result = adfuller(data_log.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Fit ARIMA model
model = ARIMA(data_log, order=(2, 1, 2))
arima_model = model.fit()

# Summary of the model
print(arima_model.summary())

# Plotting the original series, fitted values, and forecast
plt.figure(figsize=(14, 8))

# Original time series
plt.plot(data_log, label='Log of Total')

# Fitted values
plt.plot(arima_model.fittedvalues, color='red', label='Fitted Values')

# Forecast future values (e.g., next 24 months)
forecast_steps = 24
forecast = arima_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data_log.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

# Transform forecast back to original scale
forecast_original = np.exp(forecast)

# Combine original, fitted values, and forecast in the same plot
plt.plot(data_log.index, np.exp(data_log), label='Original Series')
plt.plot(arima_model.fittedvalues.index, np.exp(arima_model.fittedvalues), color='red', label='Fitted Values')
plt.plot(forecast_index, forecast_original, color='green', label='Forecast')

plt.legend()
plt.title('Original Series, Fitted Values, and Forecast')
plt.xlabel('Date')
plt.ylabel('Total')
plt.show()

# Create the ACF and PACF plots
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot ACF
plot_acf(data_log.dropna(), ax=ax[0])
ax[0].set_title('Autocorrelation Function (ACF) - Log Transformed Data')

# Plot PACF
plot_pacf(data_log.dropna(), ax=ax[1])
ax[1].set_title('Partial Autocorrelation Function (PACF) - Log Transformed Data')

plt.tight_layout()
plt.show()

# Log transformation to stabilize the variance
data_log = np.log(data['Total'])

# Fit ARIMA model
model = ARIMA(data_log, order=(2, 1, 2))
arima_model = model.fit()

# Generate forecast with confidence intervals
forecast_result = arima_model.get_forecast(steps=24, alpha=0.05)
forecast = forecast_result.predicted_mean
stderr = forecast_result.se_mean
conf_int = forecast_result.conf_int()

# Create forecast index
forecast_index = pd.date_range(start=data_log.index[-1] + pd.DateOffset(months=1), periods=24, freq='MS')

# Plotting the original series, fitted values, and forecast
plt.figure(figsize=(12, 8))
plt.plot(data_log, label='Log of Total')
plt.plot(arima_model.fittedvalues, color='red', label='Fitted Values')
plt.plot(forecast_index, forecast, color='green', label='Forecast')

# Access the confidence intervals correctly
lower_conf = conf_int.iloc[:, 0]  # Lower bound
upper_conf = conf_int.iloc[:, 1]  # Upper bound

# Plot confidence intervals
plt.fill_between(forecast_index, lower_conf, upper_conf, color='green', alpha=0.2, label='95% Confidence Interval')

plt.legend()
plt.title('Original Series, Fitted Values, and Forecast with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Log(Total)')
plt.show()

"""# **SARIMA**"""

# Seasonal differencing (for monthly data with annual seasonality, period=12)
data['Seasonal_Difference'] = data['Total'].diff(12) # Changed df to data

# Dropping NA values after differencing
seasonal_diff = data['Seasonal_Difference'].dropna() # Changed df to data

# Plot ACF and PACF for the seasonally differenced data
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# ACF plot
plot_acf(seasonal_diff, lags=48, ax=ax[0])
ax[0].set_title('ACF of Seasonally Differenced Catfish Data')

# PACF plot
plot_pacf(seasonal_diff, lags=48, ax=ax[1])
ax[1].set_title('PACF of Seasonally Differenced Catfish Data')

plt.tight_layout()
plt.show()

# Define the SARIMA model
sarima_model = sm.tsa.statespace.SARIMAX(data['Total'], # Changed df to data
                                         order=(1, 1, 1),  # (p, d, q)
                                         seasonal_order=(1, 1, 1, 12),  # (P, D, Q, S) where S is the seasonal period
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)

# Fit the model
sarima_results = sarima_model.fit()

# Summary of the model
print(sarima_results.summary())

# Log-transform the 'Total' data
data['Log_Total'] = np.log(data['Total']) # Changed df to data

# Split data into training and testing sets
train_data = data['Log_Total'][:int(0.8*len(data))] # Changed df to data
test_data = data['Log_Total'][int(0.8*len(data)):] # Changed df to data

# Fit SARIMA model
model = SARIMAX(train_data, order=(4, 1, 4), seasonal_order=(4, 1, 4, 12))
sarima_model = model.fit(disp=False)

# Predicting in-sample (fitted values) and out-of-sample (forecast)
fitted_values = sarima_model.fittedvalues
forecast_steps = len(test_data)
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = test_data.index
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Total'], label='Original Series', color='blue') # Changed df to data
plt.plot(train_data.index, fitted_values, label='Fitted Values', color='red')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.2)

plt.title('SARIMA Model - Original Series and Forecast')
plt.xlabel('Date')
plt.ylabel('Log of Total')
plt.legend()
plt.show()
