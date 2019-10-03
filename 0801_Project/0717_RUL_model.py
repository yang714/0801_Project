import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential, Input, Model
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, LSTM, TimeDistributed
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import RUL_projectV2 as RULP
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

x_d, y_d = RULP.crear_data()
# Standardization  , if necessary
# x_d = x_d.reshape(x_d.shape[0],x_d.shape[1])
# scaler = MinMaxScaler(feature_range=(0,1))
# x_d =scaler.fit_transform(x_d)
# x_d = x_d.reshape(x_d.shape[0],x_d.shape[1],1)
x_train, x_test, y_train, y_test = train_test_split(x_d, y_d, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x = Input(shape=(x_train.shape[1:]))
# x1=Flatten()(x)
y = LSTM(units=20, return_sequences=True)(x)
print("y", y.shape)
y1 = Flatten()(y)
print("y", y.shape)
output = Dense(1)(y1)
print("output", output.shape)
# output shape: (1, 1)
model = Model(x, output)
model.compile(loss="mse", optimizer="adam", metrics=['mae'])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=50, min_lr=0.01)

hist = model.fit(x_train, y_train, batch_size=200, nb_epoch=500,
                 verbose=1, validation_data=(x_test, y_test),
                 # callbacks = [TestCallback((x_train, Y_train)), reduce_lr, keras.callbacks.TensorBoard(log_dir='./log'+fname, histogram_freq=1)])
                 callbacks=[reduce_lr])

# DIRECTORY FOR SAVING THE TRAINED MODEL
save_model = model.save("E:/Python_New_Project/NTUST/model_RUL_0801_ver2.h5")

#
# prediction = model.(x_test)
# print(prediction[:10])
# print(prediction.shape)
# print(y_test.shape)
