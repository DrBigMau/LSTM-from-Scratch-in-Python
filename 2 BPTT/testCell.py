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

