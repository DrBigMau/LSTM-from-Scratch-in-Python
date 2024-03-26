# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 04:25:28 2024

@author: MMH_user
"""

import numpy as np

class LSTM():
    
    def __init__(self, n_neurons):
        
        self.n_neurons = n_neurons
        
        #forget gate
        self.Uf = 0.1*np.random.rand(n_neurons, 1)
        self.bf = 0.1*np.random.rand(n_neurons, 1)
        self.Wf = 0.1*np.random.rand(n_neurons, n_neurons)
        
        #input gate
        self.Ui = 0.1*np.random.rand(n_neurons, 1)
        self.bi = 0.1*np.random.rand(n_neurons, 1)
        self.Wi = 0.1*np.random.rand(n_neurons, n_neurons)
        
        #output gate
        self.Uo = 0.1*np.random.rand(n_neurons, 1)
        self.bo = 0.1*np.random.rand(n_neurons, 1)
        self.Wo = 0.1*np.random.rand(n_neurons, n_neurons)
        
        #c tilde
        self.Ug = 0.1*np.random.rand(n_neurons, 1)
        self.bg = 0.1*np.random.rand(n_neurons, 1)
        self.Wg = 0.1*np.random.rand(n_neurons, n_neurons)
        
        
    def forward(self, X_t):
         
        T = max(X_t.shape)
        
        self.T   = T
        self.X_t = X_t
        
        n_neurons = self.n_neurons
        
        self.H       = [np.zeros((n_neurons, 1)) for t in range(T+1)]
        self.C       = [np.zeros((n_neurons, 1)) for t in range(T+1)]
        self.C_tilde = [np.zeros((n_neurons, 1)) for t in range(T)]
        
        self.F       = [np.zeros((n_neurons, 1)) for t in range(T)]
        self.O       = [np.zeros((n_neurons, 1)) for t in range(T)]
        self.I       = [np.zeros((n_neurons, 1)) for t in range(T)]
        
        #forget gate
        self.dUf = 0.1*np.random.rand(n_neurons, 1)
        self.dbf = 0.1*np.random.rand(n_neurons, 1)
        self.dWf = 0.1*np.random.rand(n_neurons, n_neurons)
        
        #input gate
        self.dUi = 0.1*np.random.rand(n_neurons, 1)
        self.dbi = 0.1*np.random.rand(n_neurons, 1)
        self.dWi = 0.1*np.random.rand(n_neurons, n_neurons)
        
        #output gate
        self.dUo = 0.1*np.random.rand(n_neurons, 1)
        self.dbo = 0.1*np.random.rand(n_neurons, 1)
        self.dWo = 0.1*np.random.rand(n_neurons, n_neurons)
        
        #c tilde
        self.dUg = 0.1*np.random.rand(n_neurons, 1)
        self.dbg = 0.1*np.random.rand(n_neurons, 1)
        self.dWg = 0.1*np.random.rand(n_neurons, n_neurons)
        
        Sigmf    = [Sigmoid() for t in range(T)]
        Sigmi    = [Sigmoid() for t in range(T)]
        Sigmo    = [Sigmoid() for t in range(T)]
        
        Tanh1    = [Tanh() for t in range(T)]
        Tanh2    = [Tanh() for t in range(T)]
        
        ht       = self.H[0]
        ct       = self.C[0]
        
        #calling the LSTM cell
        [H, C, self.Sigmf, self.Sigmi, self.Sigmo, self.Tanh1, self.Tanh2, F, O, I, C_tilde]\
        = self.LSTMCell(X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2,\
                      self.H, self.C, self.F, self.O, self.I, self.C_tilde)

            
    def LSTMCell(self, X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2,\
                 H, C, F, O, I, C_tilde):
        
        for t, xt in enumerate(X_t):
            
            xt = xt.reshape(1,1)
            
            #forget gate
            outf = np.dot(self.Uf, xt) + np.dot(self.Wf, ht) + self.bf
            Sigmf[t].forward(outf)
            ft   = Sigmf[t].output
            
            #input gate
            outi = np.dot(self.Ui, xt) + np.dot(self.Wi, ht) + self.bi
            Sigmi[t].forward(outi)
            it   = Sigmi[t].output
            
            #output gate
            outo = np.dot(self.Uo, xt) + np.dot(self.Wo, ht) + self.bo
            Sigmo[t].forward(outo)
            ot   = Sigmo[t].output
            
            #c tilde
            outct_tilde = np.dot(self.Ug, xt) + np.dot(self.Wg, ht) + self.bg
            Tanh1[t].forward(outct_tilde)
            ct_tilde    = Tanh1[t].output
            
            
            ct = np.multiply(ft, ct) + np.multiply(it, ct_tilde)
            
            Tanh2[t].forward(ct)
            ht  = np.multiply(Tanh2[t].output,ot)
            
            H[t+1]     = ht
            C[t+1]     = ct
            C_tilde[t] = ct_tilde
            
            F[t] = ft
            O[t] = ot
            I[t] = it
        
        return(H, C, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2, F, O, I, C_tilde)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

###############################################################################
# different activation options
###############################################################################
class Sigmoid:
        
    def forward(self, M):
        
        sigm        = np.clip(1/(1 + np.exp(-M)), 1e-7, 1 - 1e-7)
        self.output = sigm
        self.inputs = sigm #needed for back prop
            
    def backward(self, dvalues):
        
        sigm         = self.inputs
        deriv        = np.multiply(sigm, (1 - sigm))
        self.dinputs = np.multiply(deriv, dvalues)

###############################################################################
#
###############################################################################
class Tanh:
    
    def forward(self, inputs):
        
        self.output = np.tanh(inputs)
        self.inputs = inputs
        
    def backward(self, dvalues):
        
        deriv = 1 - self.output**2
        self.dinputs = np.multiply(deriv, dvalues)
###############################################################################
#
###############################################################################
        
        
        
        
        
        
        
        
        
        
        
        
        
        