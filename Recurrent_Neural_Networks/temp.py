# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#fraud detection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Credit_Card_Applications.csv')
#splitting the data into two datsets(x:column 1-14,y:last column)
x=dataset.iloc[:,:-1].values
#we are going to use only x as its unsupervised learning
y=dataset.iloc[:,-1].values

#Feature scaling(using normalization:all the fetaures will be between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
x=sc.fit_transform(x)

#training the som
from minisom import MiniSom 
som=MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100)

#visualizing the result 
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i,j in enumerate(x):
    w=som.winner(j)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

#finding the frauds
mappings=som.win_map(x)
print(np.shape(mappings[(5,4)]))
print(np.shape(mappings[(5,2)]))
print(np.shape(mappings[(4,3)]))
frauds=np.concatenate((mappings[(5,4)],mappings[(5,2)]),axis=0)
frauds=sc.inverse_transform(frauds)



























