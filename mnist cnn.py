# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:31:49 2019

@author: pradeesh
"""

from keras.datasets import mnist
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D,Activation,Dropout,MaxPool2D,Flatten
from keras.utils import to_categorical
xtrain.shape
ytrain.shape

xtrain=xtrain.reshape(60000,28,28,1)
xtrain
xtest=xtest.reshape(10000,28,28,1)
xtrain

ytrain=to_categorical(ytrain)
ytrain
ytest=to_categorical(ytest)
ytest


xtrain=xtrain.astype('float32')
xtrain
xtest=xtest.astype('float32')
xtest
xtrain/255
xtest/255

c=Sequential()
c.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
#c.add(activation='relu')
c.add(MaxPool2D(pool_size=(2,2)))
c.add(Flatten())
c.add(Dense(units=100,kernel_initializer='uniform',activation='relu'))
c.add(Dense(units=10,activation='sigmoid'))
c.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
c.fit(xtrain,ytrain,batch_size=10,epochs=25)
x=c.predict(xtest)
print(x)


import matplotlib.pyplot as plt
plt.imshow(xtrain[1])
plt.imshow(xtest[1])
plt.show()
