# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:25:28 2020

@author: KIIT
""" 
#part-1 data preprocessing
import numpy as np
import matplotlib as plt
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






















