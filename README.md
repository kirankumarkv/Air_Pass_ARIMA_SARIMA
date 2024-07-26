
## Contributions
![image](https://github.com/user-attachments/assets/f3413c8a-4c8f-4af3-a380-404f73ab6ce9)

- **Data Preprocessing:** Includes stationarity tests (ADF test) and differencing the data.
- **Model Selection:** Determined model order using ACF and PACF plots.
  ![image](https://github.com/user-attachments/assets/5cc7e352-5d75-4281-ab98-12180d98d271)

- **Model Training and Validation:** Split data, trained ARIMA and SARIMA models, and evaluated performance.
- **Parameter Tuning:** Performed grid search for optimal ARIMA parameters.
- **Model Diagnostics:** Analyzed residuals and performed residual diagnostics.
- **Forecasting and Visualization:** Created forecasts, plotted original and fitted values, and visualized results.
- ![image](https://github.com/user-attachments/assets/aa15b73e-fc01-4daf-b3ec-70bee33a7137)

- **Advanced Techniques:** Included SARIMA modeling for seasonal data and incorporated exogenous variables.
![image](https://github.com/user-attachments/assets/a4d2a7ed-d754-43fe-bca5-76eebedef08b)

## License

This notebook is licensed under the MIT License. 

### MIT License

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Additional Code Snippets

### 1. Rolling Forecast Origin (Backtesting)

```python
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
```

### 2. Ensemble Methods

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(data_log, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)).fit()
print(sarima_model.summary())
```

### 3. Forecasting Intervals

```python
forecast_result = arima_model.get_forecast(steps=24, alpha=0.05)

forecast = forecast_result.predicted_mean
stderr = forecast_result.se_mean
conf_int = forecast_result.conf_int()

forecast_index = pd.date_range(start=data_log.index[-1], periods=24 + 1, freq='MS')[1:]

plt.figure(figsize=(12, 8))
plt.plot(data_log, label='Original Series')
plt.plot(arima_model.fittedvalues, color='red', label='Fitted Values')
plt.plot(forecast_index, forecast, color='green', label='Forecast')

lower_conf = conf_int.iloc[:, 0]
upper_conf = conf_int.iloc[:, 1]

plt.fill_between(forecast_index, lower_conf, upper_conf, color='green', alpha=0.2)
plt.legend()
plt.show()
```

### 4. SARIMA with Log Transformation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv('catfish.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data['Log_Total'] = np.log(data['Total'])

train_data = data['Log_Total'][:int(0.8*len(data))]
test_data = data['Log_Total'][int(0.8*len(data)):]

model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_model = model.fit(disp=False)

fitted_values = sarima_model.fittedvalues
forecast_steps = len(test_data)
forecast = sarima_model.get_forecast(steps=forecast_steps)
forecast_index = test_data.index
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Log_Total'], label='Original Series', color='blue')
plt.plot(train_data.index, fitted_values, label='Fitted Values', color='red')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.2)

plt.title('SARIMA Model - Original Series and Fitted Values')
plt.xlabel('Date')
plt.ylabel('Log of Total')
plt.legend()
plt.show()
