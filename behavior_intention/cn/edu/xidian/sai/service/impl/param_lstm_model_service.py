'''
Created on Nov 29, 2023

@author: 13507
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense#, Embedding, Dropout
from tensorflow.keras.layers import LSTM
#from sklearn.metrics import mean_squared_error
#from torch.nn import LSTM
#from torch.nn import Sequential
#import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 禁用所有GPU

def LSTM_model(trainx, trainy):
    # use the train
    model = Sequential()
    model.add(LSTM(100, activation="tanh", return_sequences = True, input_shape=(trainx.shape[1], trainx.shape[2])))
    model.add(LSTM(100, activation="tanh"))
    model.add(Dense(trainy.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    # model.fit(trainx, trainy, epochs = 50, verbose=2, callbacks=[callbacks])
    model.fit(trainx, trainy, epochs=50, verbose=2, batch_size=32)
    model.summary()
    return model