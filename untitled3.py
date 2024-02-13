import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

!pip install pmdarima
from pmdarima.arima import auto_arima

#Creatng sequential LSTM model
df = pd.read_csv('gold_price_data.csv')

df.info()

df.describe()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace= True)

#Plotting the data
plt.figure(figsize=(15, 6))
df['Value'].plot()
plt.ylabel("Price")
plt.xlabel("Date")
plt.title("Gold Price in 1970-2020")
plt.show()

n_cols = 1
dataset = df["Value"]
dataset = pd.DataFrame(dataset)
data = dataset.values
data.shape

min_max_scaler = MinMaxScaler(feature_range=(0, 1))
min_max_scaled_data = min_max_scaler.fit_transform(np.array(data))

train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
print("Train set size :",train_size,"Test set size :",test_size)

time_steps = 100
train_data = min_max_scaled_data[0:train_size, :]
test_data = min_max_scaled_data[train_size - time_steps:, :]
train_data.shape, test_data.shape

# Creating a training set
x_train = []
y_train = []

for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, :n_cols])
    y_train.append(train_data[i, :n_cols])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping the input to format n_samples, time_steps, n_features
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_cols))

model_lstm = Sequential([
    LSTM(100, activation='relu', return_sequences= True, input_shape=(x_train.shape[1], n_cols)),
    LSTM(64, activation='relu', return_sequences= False),
    Dense(32, activation='linear'),
    Dense(16, activation='linear'),
    Dense(n_cols)
])
opt = Adam(learning_rate=0.001)
model_lstm.compile(optimizer= opt, loss= 'mean_squared_error' , metrics= "mean_absolute_error")

model_lstm.summary()

# Fitting the LSTM to the training set
LSTM = model_lstm.fit(x_train, y_train, epochs= 100, batch_size= 32)

plt.figure(figsize=(12, 8))
plt.plot(LSTM.history["loss"])
plt.plot(LSTM.history["mean_absolute_error"])
plt.legend(['Mean Squared Error','Mean Absolute Error'])
plt.title("LSTM model losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# Creating a testing set
x_test = []
y_test = []

for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, 0:n_cols])
    y_test.append(test_data[i, 0:n_cols])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_cols))
x_test.shape , y_test.shape

predictions = model_lstm.predict(y_test)

#Standard metrics for LSTM model
mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)

#inverse scaling
predictions = min_max_scaler.inverse_transform(predictions)
y_test = min_max_scaler.inverse_transform(y_test)

pred_true = pd.DataFrame(data={'Predictions':predictions.flatten(), 'True':y_test.flatten()})

plt.figure(figsize = (16, 6))
plt.plot(pred_true['Predictions'])
plt.plot(pred_true['True'])
plt.legend(['Predictions', 'True'])
plt.show()

train = dataset.iloc[:train_size , 0:1]
test = dataset.iloc[train_size: , 0:1]
test['Predictions'] = predictions

plt.figure(figsize= (16, 6))
plt.title('Gold Price 1970-2020', fontsize= 18)
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Gold Price', fontsize= 18)
plt.plot(train['Value'], linewidth= 3)
plt.plot(test['Value'], linewidth= 3)
plt.plot(test["Predictions"], linewidth= 3)
plt.legend(['History', 'Actual', 'Predictions'])

# Creating ARIMA model for comparison
df_2 = pd.read_csv('gold_price_data.csv')
df_2['Date'] = pd.to_datetime(df_2['Date'])
df_2.set_index("Date" , inplace = True)
df_2 = df_2["Value"].resample("1D").mean().fillna(method = "ffill").to_frame()
dates = df_2.index
train_size_2 = int(len(df_2) * 0.8)

df_2["Value"] = min_max_scaler.fit_transform(df_2["Value"].values.reshape(-1, 1))
df_2 = pd.DataFrame(df_2)

data_2 = df_2["Value"]
train_2 = data_2.iloc[:train_size_2]
test_2 = data_2.iloc[train_size_2:]

model_arima = auto_arima(train_2, stepwise = False, seasonal = False, stationary = False)

model_arima.summary()

model_arima.fit(train_2)

#Basic metrics for ARIMA model
predictions_2 = model_arima.predict(n_periods = len(test_2))
mean_squared_error(test_2, predictions_2), mean_absolute_error(test_2 , predictions_2)

#Inverse scaling data
pred_arima = min_max_scaler.inverse_transform(predictions_2.values.reshape(-1, 1))
test_arima = min_max_scaler.inverse_transform(test_2.values.reshape(-1, 1))
train_arima = min_max_scaler.inverse_transform(train_2.values.reshape(-1, 1))

pred_true_2 = pd.DataFrame({"True": test_arima.flatten(), "Predictions": pred_arima.flatten()})

plt.figure(figsize = (16, 6))
plt.plot(pred_true_2['True'])
plt.plot(pred_true_2['Predictions'])
plt.legend(['True', 'Predictions'])
plt.show()

plt.figure(figsize = (16, 6))
plt.plot(df_2.index[:train_size_2] , train_arima, color='red', label='History')
plt.plot(df_2.index[train_size_2:] , test_arima , color='orange', label='Actual')
plt.plot(df_2.index[train_size_2:] , pred_arima , color='green', label='Prediction')
plt.xlabel('Time')
plt.ylabel('Data')
plt.title('Gold Price 1970-2020', fontsize=18)
plt.legend()
plt.show()
