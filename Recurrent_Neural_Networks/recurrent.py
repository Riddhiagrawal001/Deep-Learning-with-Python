# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:25:28 2020

@author: KIIT
""" 
#part-1 data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
training_set=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=training_set.iloc[:,1:2].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)

#getting the inputs and  outputs
x_train=training_set[0:1257]
y_train=training_set[1:1258]

#reshaping
x_train=np.reshape(x_train, (1257,1,1))

#part-2 Building the rnn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#initialing the rnn
regressor=Sequential()

#adding the input and lstm layer
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))

#adding the output layer
regressor.add(Dense(units=1))

#compile the rnn
regressor.compile(optimizer='adam',loss='mean_squared_error')

#fitting thr rnn to the training set
regressor.fit(x_train,y_train,batch_size=32,epochs=200)

#part-3 making the predictions and visualizing the result
test_set=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=test_set.iloc[:,1:2].values

#getting the predicted stock price
inputs=real_stock_price
inputs=sc.transform(inputs)
inputs=np.reshape(inputs, (20,1,1))
predicted_stock_price=regressor.predict(inputs)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#visualizing the result
plt.plot(real_stock_price,color='red',label='Real google stock price')
plt.plot(real_stock_price,color='blue',label='Predicted google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

#homework solution
#getting the real stock price of 12-16
real_stock_train=pd.read_csv("Google_Stock_Price_Train.csv")
real_stock_train=real_stock_train.iloc[:,1:2].values

#getting the predicted data
predicted_stock_price_train=regressor.predict(x_train)
predicted_stock_price_train=sc.inverse_transform(predicted_stock_price_train)

#visualizing the data
plt.plot(real_stock_train,color='red',label='Real google stock price')
plt.plot(predicted_stock_price_train,color='blue',label='Predicted google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

#part-4 evaluationg the rnn
import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))




