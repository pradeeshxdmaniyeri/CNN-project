# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:13:22 2019

@author: pradeesh
"""

from keras.datasets import fashion_mnist
(xtrain,ytrain),(xtest,ytest)=fashion_mnist.load_data()
#fashion_mnist
(xtrain,ytrain),(xtest,ytest)
xtrain.shape
xtest.shape
ytrain.shape
from keras import Sequential
from keras.layers import Dense
from keras.layers import MaxPool2D,Conv2D,Flatten,Dropout,Activation
from keras.utils import to_categorical

xtrain=xtrain.reshape(60000,28,28,1)
xtrain
xtest=xtest.reshape(10000,28,28,1)
xtest

ytrain=to_categorical(ytrain)
ytrain
ytest=to_categorical(ytest)
ytest

xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')
xtrain/255
xtest/255
xtrain
xtest

c=Sequential()
c.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
c.add(MaxPool2D(pool_size=(5,5)))
c.add(Conv2D(64,(3,3),padding='same',activation='relu'))
c.add(MaxPool2D(pool_size=(2,2)))
c.add(Flatten())
c.add(Dense(units=100,kernel_initializer='uniform',activation='relu'))
c.add(Dense(units=10,kernel_initializer='uniform',activation='sigmoid'))
c.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
c.fit(xtrain,ytrain,batch_size=10,epochs=2)
a=c.predict(xtest)
print(a)

from sklearn.metrics import r2_score
r=r2_score(ytest,a)
r


import matplotlib.pyplot as plt
plt.imshow(xtrain[5])
plt.show()
