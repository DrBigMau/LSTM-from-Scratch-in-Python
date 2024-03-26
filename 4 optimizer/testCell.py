# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:30:41 2024

@author: MMH_user
"""
import numpy as np
import matplotlib.pyplot as plt

X_t = np.arange(-70,10,0.1)
#X_t = np.arange(-10,10,0.1)
X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1) + np.exp((0.5*X_t + 20)*0.05)

plt.plot(X_t, Y_t)
plt.show()

from LSTM import *

n_neurons = 200
lstm      = LSTM(n_neurons)
T         = max(X_t.shape)
dense1    = Layer_Dense(n_neurons,T)
dense2    = Layer_Dense(T,1)

optimizer_lstm = Optimizer_SGD_LSTM()
optimizer      = Optimizer_SGD()

n_epoch = 100
Monitor = np.zeros((100))

for n in range(n_epoch):
    
    lstm.forward(X_t)
    H = np.array(lstm.H)
    H = H.reshape((H.shape[0], H.shape[1]))
    
    dense1.forward(H[1:,:])
    dense2.forward(dense1.output)
    
    Y_hat = dense2.output
    
    dY = Y_hat - Y_t
    
    L = float(0.5* np.dot(dY.T,dY)/T)
    
    Monitor[n] = L
    
    dense2.backward(dY)
    dense1.backward(dense2.dinputs)
    
    lstm.backward(dense1.dinputs)
    
    optimizer_lstm.pre_update_params()
    optimizer.pre_update_params()
    
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    
    optimizer_lstm.update_params(lstm)
    
    optimizer_lstm.post_update_params()
    optimizer.post_update_params()
    
    
    print(f'current MSSE = {L: 0.3f}')

plt.plot(range(n_epoch),Monitor)
plt.xlabel('epochs')
plt.ylabel('MSSE')
plt.yscale('log')

















