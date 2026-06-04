# Time Series

---

## What Makes Time Series Special?
- Data points are **ordered in time** — order matters
- Can't randomly shuffle (breaks temporal order)
- Train/test split must respect time: always train on PAST, test on FUTURE
- Has patterns: trend, seasonality, cycles, noise

---

## Key Concepts

### Components of a Time Series
```
Trend:       long-term increase/decrease
Seasonality: regular pattern (daily/weekly/yearly)
Cyclical:    irregular longer-term waves (economic cycles)
Noise:       random variation

Additive:       y = Trend + Seasonality + Noise  (seasonality constant size)
Multiplicative: y = Trend × Seasonality × Noise  (seasonality grows with trend)
```

### Stationarity
> A stationary series has constant mean, variance, and autocorrelation over time.  
> Most models require stationarity.

```python
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller test
result = adfuller(df['value'])
print(f"ADF Statistic: {result[0]:.3f}")
print(f"p-value: {result[1]:.3f}")
# p < 0.05 → stationary
# p > 0.05 → NOT stationary → need differencing

# Make stationary: differencing
df['diff'] = df['value'].diff()          # first difference
df['diff2'] = df['value'].diff().diff()  # second difference
df['log'] = np.log(df['value'])          # log transform (for multiplicative)
```

### Autocorrelation
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['value'], lags=40)   # ACF: correlation with past values
plot_pacf(df['value'], lags=40)  # PACF: direct correlation (removing intermediate)
# Use these to pick ARIMA parameters
```

---

## Train/Test Split for Time Series

```python
# NEVER use random split!
train_size = int(len(df) * 0.8)
train = df[:train_size]
test  = df[train_size:]

# Time series cross-validation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
```

---

## ARIMA

**AR** (AutoRegressive): uses past values to predict  
**I** (Integrated): differencing to make stationary  
**MA** (Moving Average): uses past errors to predict  

```
ARIMA(p, d, q)
p = AR order   (how many past values)
d = differencing order  (usually 0 or 1)
q = MA order   (how many past errors)

Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q,m)  where m = season length
```

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(2, 1, 2))  # AR=2, diff=1, MA=2
result = model.fit()
print(result.summary())

# Forecast
forecast = result.forecast(steps=len(test))

# Auto-find best parameters
from pmdarima import auto_arima
auto_model = auto_arima(train, seasonal=True, m=12,  # m=12 for monthly
                         stepwise=True, suppress_warnings=True)
print(auto_model.summary())
forecast = auto_model.predict(len(test))
```

---

## Prophet (by Meta — Easy & Powerful)

```python
from prophet import Prophet
import pandas as pd

# Prophet requires columns: 'ds' (datetime) and 'y' (value)
df_prophet = df.rename(columns={'date': 'ds', 'sales': 'y'})

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05  # flexibility of trend
)

# Add custom seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Add holidays
from prophet.make_holidays import make_holidays_df
model.add_country_holidays(country_name='IN')

model.fit(df_prophet)

# Future dataframe
future = model.make_future_dataframe(periods=30)  # forecast 30 days
forecast = model.predict(future)

# Visualize
model.plot(forecast)
model.plot_components(forecast)  # trend + seasonality breakdown
```

---

## LSTM for Time Series

```python
import numpy as np
from tensorflow import keras

# Create sequences (sliding window)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(train.values.reshape(-1,1))

seq_len = 30
X_train, y_train = create_sequences(data_scaled, seq_len)

# Model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
```

---

## ML Approach (LightGBM with Features — Best in Practice)

```python
# Create lag and rolling features
df['lag_1']    = df['sales'].shift(1)
df['lag_7']    = df['sales'].shift(7)
df['lag_30']   = df['sales'].shift(30)
df['roll_7']   = df['sales'].shift(1).rolling(7).mean()
df['roll_30']  = df['sales'].shift(1).rolling(30).mean()

# Date features
df['dayofweek'] = df['date'].dt.dayofweek
df['month']     = df['date'].dt.month
df['is_weekend']= df['dayofweek'].isin([5,6]).astype(int)

# Train/test split (by time!)
train = df[df['date'] < '2024-01-01']
test  = df[df['date'] >= '2024-01-01']

import lightgbm as lgb
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
model.fit(train[features], train['sales'])
preds = model.predict(test[features])
```

---

## Time Series Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
```

---

## When to Use What

| Situation | Model |
|-----------|-------|
| Short series, interpretability | ARIMA / auto_arima |
| Business forecasting, holidays | Prophet |
| Many related series, tabular features | LightGBM with lag features |
| Very long sequences, complex patterns | LSTM / Transformer |
| Quick baseline | Previous value (naive) or rolling average |
