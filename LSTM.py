import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt

# Load data set and split training set and test set
index = 12701
train_number = 2000
test_number = 500
data_total = pd.read_csv('price.csv')
print(data_total.iloc[index])
# split training set
save_data = data_total.iloc[index: index + train_number]
file_name = 'train' + '.csv'
save_data.to_csv(file_name, index=False)
# split test set
save_data = data_total.iloc[index + train_number: index + train_number+test_number]
file_name = 'test' + '.csv'
save_data.to_csv(file_name, index=False)

# Load training set
train = pd.read_csv('train.csv')
training_set = train.iloc[:train_number, 5:6].values

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create 3D data using timestep
X_train = []
Y_train = []
for i in range(60, train_number):
    X_train.append(training_set_scaled[i-60:i, 0])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Create and train simple LSTM model with one layer
# model = Sequential()
# model.add(LSTM(units=1, input_shape=(X_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.compile(optimizer='Adam', loss='mean_squared_error')
# model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Create and train stacked LSTM Model with four layers (Train)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='Adam', loss='mean_squared_error')
# # different optimizer:
# model.compile(optimizer='SGD', loss='mean_squared_error')
# model.compile(optimizer='RMSProp', loss='mean_squared_error')
# model.compile(optimizer='Adagrad', loss='mean_squared_error')


model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Load test set and make prediction (Test)
test = pd.read_csv('test.csv')
test_set = test.iloc[:test_number, 5:6].values
total_data = np.vstack((training_set[len(training_set)-60:], test_set))
total_data = sc.transform(total_data)
X_test = []
for i in range(60, test_number+60):
    X_test.append(total_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

# Evaluation
print('MAE: ', mean_absolute_error(test_set, predicted_price))
print('RMSE: ', sqrt(mean_squared_error(test_set, predicted_price)))
print('R2 score: ', r2_score(test_set, predicted_price))

# # Show the output without previous price
# plt.plot(test_set, color='black', label='Real price')
# plt.plot(predicted_price, color='red', label='Predicted price')
# plt.title('Price Prediction Using LSTM')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

# Show the output with previous price
X = np.linspace(0, train_number+test_number-1, train_number+test_number)
X1 = X[:train_number]
X2 = X[train_number:]
plt.plot(X1, training_set, color='black', label='Training price')
plt.plot(X2, test_set, color='blue', label='Real price')
plt.plot(X2, predicted_price, color='red', label='Predicted price')
plt.title('Price Prediction Using LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
