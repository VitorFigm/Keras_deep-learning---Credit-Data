from base import *
import numpy as np
#preprocessing
from sklearn.impute import SimpleImputer
X = SimpleImputer().fit_transform(credit_Data.drop(['c#default','i#clientid'],1))
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

Y = credit_Data['c#default']

#split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y)


#keras model
from keras.models import Sequential
from keras.layers import Dense

network = Sequential()
network.add(Dense(units=2,activation='relu',input_dim=3))
network.add(Dense(units=2,activation='relu'))
network.add(Dense(units=2,activation='relu'))

network.add(Dense(units=1,activation='sigmoid'))
network.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
network.fit(X_train,Y_train,batch_size=10,epochs=100)

scr = network.evaluate(X_test,Y_test)
print(scr)