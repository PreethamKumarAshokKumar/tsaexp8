# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING

### Name:ASHOK KUMAR PREETHAM KUMAR
### Register No: 212224040032
### Date: 06/05/2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

#### import library functions

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
```
#### Read the AirPassengers dataset

```
data = pd.read_csv('AirPassengers.csv')
```

#### Take the '#Passengers' column

```
passengers_data = data[['#Passengers']]
```

#### Display the shape and the first 10 rows of the dataset

```
print("Shape of the dataset:", passengers_data.shape)
print("First 10 rows of the dataset:")
print(passengers_data.head(10))
```

#### Plot Original Dataset (#Passengers Data)

```
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original #Passengers Data')
plt.title('Original Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
```

#### Moving Average Perform rolling average transformation with a window size of 5 and 10

```
rolling_mean_5 = passengers_data['#Passengers'].rolling(window=5).mean()
rolling_mean_10 = passengers_data['#Passengers'].rolling(window=10).mean()
```

#### Display the first 10 and 20 vales of rolling means with window sizes 5 and 10 respectively

```
rolling_mean_5.head(10)
rolling_mean_10.head(20)
```

#### Plot Moving Average

```
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['#Passengers'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Passenger Data')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid()
plt.show()
```

#### Perform data transformation to better fit the model

```
data_monthly = data.resample('MS').sum()   #Month start
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),index=data.index)
```

#### Exponential Smoothing
```
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, yes even zeros
x=int(len(scaled_data)*0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()
```

#### Make predictions for one fourth of the data
```
model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
predictions.plot(ax=ax)
ax.legend(["data_monthly", "predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')
```


### OUTPUT:

#### Original Data

![image](https://github.com/user-attachments/assets/148f8949-df21-498b-9263-9826ca09ddb3)

![image](https://github.com/user-attachments/assets/612f8b4c-0eff-40df-a137-0a472fa5d12a)

#### Moving Average

![image](https://github.com/user-attachments/assets/e0fdd5be-0418-49d3-bce0-af5a6ac47454)

#### Plot of Moving Average Data

![image](https://github.com/user-attachments/assets/5cc908ca-7162-4200-bfd0-bda574047357)

### Exponential Smoothing
#### Test

![image](https://github.com/user-attachments/assets/f6eef47e-08e6-4e32-b308-4855f167f66f)

#### Prediction

![image](https://github.com/user-attachments/assets/ddcf4f0a-b00b-41ab-9437-64baafc1854b)



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
