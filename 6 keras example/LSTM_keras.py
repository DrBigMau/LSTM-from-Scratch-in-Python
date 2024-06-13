# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 22:00:50 2024

@author: MMH_user
"""

#   modyfied example from Jason Brownlee 
#  "Multivariate Time Series Forecasting with LSTMs in Keras" 
#   https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
 

###############################################################################
#####usage#####################################################################
# =============================================================================
# from LSTM_keras import LSTM_keras
# L = LSTM_keras()
# L.CreateModel(dt = 40, n_neurons = 150) #multiple time lag dt > 1
# L.Predict()
# 
# =============================================================================
###############################################################################

### loading libraries & packages ##############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%
from os import *
from math import sqrt
#from math import factorial
from numpy import concatenate
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
###############################################################################

###############################################################################
##############auxillaries######################################################
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

#turning the raw data into more readable csv 
def RawToCSV(my_rawdata = 'raw.txt'):

    dataset = pd.read_csv(my_rawdata,\
                          parse_dates = [['year', 'month', 'day', 'hour']],\
                          index_col   = 0,\
                          date_parser = parse)
    dataset.drop('No', axis = 1, inplace = True)
    
# manually specify column names
    dataset.columns    = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    dataset['pollution'].fillna(0, inplace=True)# mark all NA values with 0
    dataset = dataset[24:]# drop the first 24 hours
    print(dataset.head(5))# summarize first 5 rows
# save to file
    dataset.to_csv('pollution.csv')

#plot data
def PlotData(groups, wind = 'No', values = None):
    
    if (values is None):
        dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
        values = dataset.values
        cols   = dataset.columns
    else:
        cols   = values.columns
        
            
    if wind != 'No':
        #just to check for plt
        WindSpdTot     = values['wnd_spd [SE]'] + values['wnd_spd [NE]'] +\
                         values['wnd_spd [NW]'] + values['wnd_spd [cv]']
            
    values = np.array(values)#making sure we have the right format for plotting
    
    
    i = 1
    # plot each column
    plt.figure(figsize = (10, 2*len(groups)))
    # specify columns to plot
    for group in groups:
        plt.subplot(len(groups), 1, i,)
        #plotting one-week moving average
        y_smooth = savgol_filter(values[:,group], 24*7, 5)
        plt.plot(y_smooth, c = 'black')
        plt.plot(values[:, group], c = [215/255, 0, 64/255, 0.5])
        if wind != 'No':
            if group in [4, 5, 6, 7]:
                #plotting total wind speed for comparison
                plt.plot(WindSpdTot, c = [0, 71/255, 171/255, 0.2])
        plt.title(cols[group], y = 0.75, loc = 'right', fontsize = 12)
        i += 1
    plt.show()
    
    
def DoTimeLag(Data, dt):
    
    Y = Data[dt:, 0]
    Y = Y.reshape((len(Y), 1))
    X = Data[dt:, 1:]
    
    for i in reversed(range(dt)):
        X = concatenate((X, Data[i:-dt+i, 1:]), axis = 1)
        
    Data = concatenate((Y,X), axis = 1)
        
    return Data
    

#normalize data to [0, 1] & one-hot encoding categoricals (e. g. wind direction)
def NormAndScale():
    
    dataset        = pd.read_csv('pollution.csv', header = 0, index_col = 0)
    values         = dataset.values
    values_new     = np.hstack((values[:,:4], values[:,5:]))
    values_new     = values_new.astype('float32')
    
    #normalizing BEFORE we one-hot encode wind directions. Otherwise 
    #information of total wind speed would get lost (each direction normalized
    #to one separately)
    scaler         = MinMaxScaler(feature_range=(0, 1)) #normalize features
    scaled         = scaler.fit_transform(values_new)
    
    #one-hot encoding wind direction
    encoder        = LabelEncoder() #integer encode direction
    WindDir        = encoder.fit_transform(values[:,4])
    Ndirections    = np.max(WindDir) + 1
    WindDirOne     = np.eye(Ndirections)[WindDir]#one hot for wind direction
    WindSpdNtimes  = np.tile(scaled[:,4],(Ndirections,1))
    WindSpdOneHot  = np.multiply(WindDirOne,WindSpdNtimes.transpose())
    scaled_new     = np.hstack((scaled[:,:4], WindSpdOneHot, scaled[:,5:]))
    
    scaled_new     = scaled_new.astype('float32') #ensure all data is float

    
    #overwriding old values with scaled/normalized
    values         = pd.DataFrame(scaled_new)
    values.columns = ['pollution', 'dew', 'temp', 'press',\
                      'wnd_spd [SE]','wnd_spd [NW]', 'wnd_spd [NE]',\
                      'wnd_spd [cv]', 'snow', 'rain']


    print(values.head())
    
    PlotData(groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], wind = 'yes',\
             values = values)
        
    return(values_new, scaled_new, scaler)

def invNormAndScale(X, model, scaler):
    
    Yhat     = model.predict(X)
    
    X        = X[:,0,:]
    
    #merging wind direction
    Wind     = np.sum(X[:,4:8], axis = 1)
    Wind     = Wind.reshape((len(Wind),1))
    
    X_new    = np.hstack((X[:,:4], Wind, X[:,8:]))
    
    #invert scaling for forecast
    inv_yhat = concatenate((Yhat, X_new), axis = 1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    
    return(inv_yhat)

###############################################################################
###############################################################################

class LSTM_keras():

    def __init__(self):
        
        #searching for preprocessed data pollution.csv. 
        #If not --> create pollution.csv
        Files = listdir()
        
        if not 'pollution.csv' in Files:
            RawToCSV()
        
        PlotData(groups = [0, 1, 2, 3, 5, 6, 7])
        print('Plotting Data...')
        
        print('Scaling & Normalizing Data...')
        [self.OrigData, self.Data, self.scaler] = NormAndScale()
        print('Plotting Normalized Data...')
        

    def CreateModel(self, n_divide = 80, n_neurons = 250, dt = 1,\
                          n_epochs = 150, learning_rate = 0.001,\
                          momentum = 0.4, opt = 'sgd'):
        
        print('Performing Training...')
        
        #calling normalized and scaled data
        Data = self.Data
        
        if dt > 1:#multiple time lag is treated as dt times extra features
            Data = DoTimeLag(Data, dt - 1)
        
        n_divide = round(len(Data[:,0])*n_divide/100)
        
        #dividing data into test and training set
        train = Data[:n_divide, :]
        test  = Data[n_divide:, :]
        
        #split into input and outputs
        train_X, train_Y = train[:, 1:], train[:, 0]
        test_X,  test_Y  = test[:, 1:], test[:, 0]
        
        #reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], dt,\
                                   int(train_X.shape[1]/dt)))
        test_X  = test_X.reshape((test_X.shape[0], dt,\
                                  int(test_X.shape[1]/dt)))
            
        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
 
        #design network: n_neurons stands for size of hidden layer
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape = (train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        
        if opt == 'sgd':
            Opt = optimizers.SGD(learning_rate = learning_rate,\
                                 momentum = momentum)
        
        if opt == 'adam':
            Opt = optimizers.Adam(learning_rate = learning_rate)
            
        
        model.compile(loss = 'mae', optimizer = Opt)
        # fit network
        history = model.fit(train_X, train_Y, epochs = n_epochs,\
                            batch_size = 72,\
                            validation_data = (test_X, test_Y),\
                            verbose = 2, shuffle = False)
        #plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
        print('...Training Done!')
        
        self.dt       = dt
        self.model    = model
        self.test_X   = test_X
        self.test_Y   = test_Y
        self.train_X  = train_X
        self.train_Y  = train_Y
        self.n_divide = n_divide
        
        
    def Predict(self):
        
        YOrig    = self.OrigData[:,0]
        
        #scaled X we need for the inverse scaler
        #make sure to have correct shape, if dt > 1
        test_X       = self.test_X
        train_X      = self.train_X
        
        #predicing and rescaling data
        Yhat_test  = invNormAndScale(test_X, self.model, self.scaler)
        Yhat_train = invNormAndScale(train_X, self.model, self.scaler)
        
        #calculate RMSE
        #rmse = sqrt(mean_squared_error(YOrig[self.dt-1+self.n_divide:],\
        #                               Yhat_test))
        #print('Test RMSE: %.3f' % rmse)
        
        fittedY = np.hstack((Yhat_train, Yhat_test))       
        All     = np.hstack((fittedY, YOrig))
        
        
        #plotting final result: actual data vs prediction (training and test)
        M = np.max(All)
        m = np.min(All)
        
        plt.plot(fittedY, c = [215/255, 0, 64/255, 0.8])
        plt.plot(YOrig,   c = [0, 71/255, 171/255, 0.2])
        plt.xlabel('time [hrs]')
        plt.ylabel('pollution')
        plt.legend(['$\hat{y}$','y'], loc = 'upper left')
        plt.fill_between([self.n_divide, len(fittedY)], m, M, color = 'k',\
                         alpha = 0.1)
        plt.title('prediction', y = 0.75, loc = 'right', fontsize = 12)
        plt.show()
        
        plt.plot(savgol_filter(fittedY,24*7,5), c = [215/255, 0, 64/255, 0.8])
        plt.plot(savgol_filter(YOrig,24*7,5), c = [0, 71/255, 171/255, 0.2])
        plt.xlabel('time [hrs]')
        plt.ylabel('pollution [weekly avg]')
        plt.legend(['$\hat{y}$','y'], loc = 'upper left')
        plt.fill_between([self.n_divide, len(fittedY)], m, M, color = 'k',\
                         alpha = 0.1)
        plt.title('prediction', y = 0.75, loc = 'right', fontsize = 12)
        plt.show()
        
        