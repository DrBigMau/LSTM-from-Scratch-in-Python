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
        [H, C, Sigmf, Sigmi, Sigmo, Tanh1,Tanh2, F, O, I, C_tilde]\
        = self.LSTMCell(X_t, ht, ct, Sigmf, Sigmi, Sigmo, Tanh1, Tanh2,\
                      self.H, self.C, self.F, self.O, self.I, self.C_tilde)
        
        
        self.F       = F
        self.O       = O
        self.I       = I
        self.C_tilde = C_tilde

        self.H       = H
        self.C       = C
        
        self.Sigmf   = Sigmf
        self.Sigmi   = Sigmi
        self.Sigmo   = Sigmo
        self.Tanh1   = Tanh1
        self.Tanh2   = Tanh2
        
            
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
    
    
    def backward(self, dvalues):
        
        #dht = dinputs from the dense layer
        
        T       = self.T
        H       = self.H
        C       = self.C
        
        O       = self.O
        I       = self.I
        C_tilde = self.C_tilde
        
        X_t     = self.X_t
        
        Sigmf   = self.Sigmf
        Sigmi   = self.Sigmi
        Sigmo   = self.Sigmo
        Tanh1   = self.Tanh1
        Tanh2   = self.Tanh2
        
        dht     = dvalues[-1,:].reshape(self.n_neurons,1)
        
        #actual BPTT
        for t in reversed(range(T)):
            
            xt = X_t[t].reshape(1,1)
            
            Tanh2[t].backward(dht)
            dtanh2 = Tanh2[t].dinputs
            
            #np.multiply, not np.dot because it was a element wise 
            #multiplication in the forward part
            dhtdtanh     = np.multiply(O[t], dtanh2)
            
            dctdft       = np.multiply(dhtdtanh,C[t-1])
            dctdit       = np.multiply(dhtdtanh,C_tilde[t])
            dctdct_tilde = np.multiply(dhtdtanh,I[t])
            
            Tanh1[t].backward(dctdct_tilde)
            dtanh1 = Tanh1[t].dinputs
            
            Sigmf[t].backward(dctdft)
            dsigmf = Sigmf[t].dinputs
            
            Sigmi[t].backward(dctdit)
            dsigmi = Sigmi[t].dinputs
            
            Sigmo[t].backward(np.multiply(dht, Tanh2[t].output))
            dsigmo = Sigmo[t].dinputs
            
            dsigmfdUf = np.dot(dsigmf,xt)
            dsigmfdWf = np.dot(dsigmf,H[t-1].T)
            
            self.dUf += dsigmfdUf
            self.dWf += dsigmfdWf
            self.dbf += dsigmf
            
            dsigmidUi = np.dot(dsigmi,xt)
            dsigmidWi = np.dot(dsigmi,H[t-1].T)
            
            self.dUi += dsigmidUi
            self.dWi += dsigmidWi
            self.dbi += dsigmi
            
            dsigmodUo = np.dot(dsigmo,xt)
            dsigmodWo = np.dot(dsigmo,H[t-1].T)
            
            self.dUo += dsigmodUo
            self.dWo += dsigmodWo
            self.dbo += dsigmo
            
            dtanh1dUg = np.dot(dtanh1,xt)
            dtanh1dWg = np.dot(dtanh1,H[t-1].T)
            
            self.dUg += dtanh1dUg
            self.dWg += dtanh1dWg
            self.dbg += dtanh1
            

            dht = np.dot(self.Wf, dsigmf) + np.dot(self.Wi, dsigmi) +\
                  np.dot(self.Wo, dsigmo) + np.dot(self.Wg, dtanh1) +\
                  dvalues[t-1,:].reshape(self.n_neurons,1)

        self.H = H
            
###############################################################################
# dense layer
###############################################################################
class Layer_Dense():
    
    def __init__(self, n_inputs, n_neurons):
        #note: we are using randn here in order to see if neg values are 
        #clipped by the ReLU
        #import numpy as np
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        
#passing on the dot product as input for the next layer, as before
    def forward(self, inputs):
        self.output  = np.dot(inputs, self.weights) + self.biases
        self.inputs  = inputs#we're gonna need for backprop
        
    def backward(self, dvalues):
        #gradients
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs  = np.dot(dvalues, self.weights.T)
        
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
class Optimizer_SGD:
    #initializing with a default learning rate of 0.1
    def __init__(self, learning_rate = 1e-5, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
                
            #now the momentum parts
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases  += bias_updates
        
    def post_update_params(self):
        self.iterations += 1
        
###############################################################################
#
###############################################################################
class Optimizer_SGD_LSTM:
    #initializing with a default learning rate of 0.1
    def __init__(self, learning_rate = 1e-5, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'Uf_momentums'):
                layer.Uf_momentums = np.zeros_like(layer.Uf)
                layer.Ui_momentums = np.zeros_like(layer.Ui)
                layer.Uo_momentums = np.zeros_like(layer.Uo)
                layer.Ug_momentums = np.zeros_like(layer.Ug)
                
                layer.Wf_momentums = np.zeros_like(layer.Wf)
                layer.Wi_momentums = np.zeros_like(layer.Wi)
                layer.Wo_momentums = np.zeros_like(layer.Wo)
                layer.Wg_momentums = np.zeros_like(layer.Wg)
                
                layer.bf_momentums = np.zeros_like(layer.bf)
                layer.bi_momentums = np.zeros_like(layer.bi)
                layer.bo_momentums = np.zeros_like(layer.bo)
                layer.bg_momentums = np.zeros_like(layer.bg)
                

            #now the momentum parts
            Uf_updates = self.momentum * layer.Uf_momentums - \
                self.current_learning_rate * layer.dUf
            layer.Uf_momentums = Uf_updates
            
            Ui_updates = self.momentum * layer.Ui_momentums - \
                self.current_learning_rate * layer.dUi
            layer.Ui_momentums = Ui_updates
            
            Uo_updates = self.momentum * layer.Uo_momentums - \
                self.current_learning_rate * layer.dUo
            layer.Uo_momentums = Uo_updates
            
            Ug_updates = self.momentum * layer.Ug_momentums - \
                self.current_learning_rate * layer.dUg
            layer.Ug_momentums = Ug_updates
            
            Wf_updates = self.momentum * layer.Wf_momentums - \
                self.current_learning_rate * layer.dWf
            layer.Wf_momentums = Wf_updates
            
            Wi_updates = self.momentum * layer.Wi_momentums - \
                self.current_learning_rate * layer.dWi
            layer.Wi_momentums = Wi_updates
            
            Wo_updates = self.momentum * layer.Wo_momentums - \
                self.current_learning_rate * layer.dWo
            layer.Wo_momentums = Wo_updates
            
            Wg_updates = self.momentum * layer.Wg_momentums - \
                self.current_learning_rate * layer.dWg
            layer.Wg_momentums = Wg_updates
            
            bf_updates = self.momentum * layer.bf_momentums - \
                self.current_learning_rate * layer.dbf
            layer.bf_momentums = bf_updates
            
            bi_updates = self.momentum * layer.bi_momentums - \
                self.current_learning_rate * layer.dbi
            layer.bi_momentums = bi_updates
            
            bo_updates = self.momentum * layer.bo_momentums - \
                self.current_learning_rate * layer.dbo
            layer.bo_momentums = bo_updates
            
            bg_updates = self.momentum * layer.bg_momentums - \
                self.current_learning_rate * layer.dbg
            layer.bg_momentums = bg_updates
            
        else:
            
            Uf_updates = -self.current_learning_rate * layer.dUf
            Ui_updates = -self.current_learning_rate * layer.dUi
            Uo_updates = -self.current_learning_rate * layer.dUo
            Ug_updates = -self.current_learning_rate * layer.dUg
            
            Wf_updates = -self.current_learning_rate * layer.dWf
            Wi_updates = -self.current_learning_rate * layer.dWi
            Wo_updates = -self.current_learning_rate * layer.dWo
            Wg_updates = -self.current_learning_rate * layer.dWg
            
            bf_updates = -self.current_learning_rate * layer.dbf
            bi_updates = -self.current_learning_rate * layer.dbi
            bo_updates = -self.current_learning_rate * layer.dbo
            bg_updates = -self.current_learning_rate * layer.dbg
            
        
        layer.Uf += Uf_updates 
        layer.Ui += Ui_updates 
        layer.Uo += Uo_updates 
        layer.Ug += Ug_updates 
        
        layer.Wf += Wf_updates 
        layer.Wi += Wi_updates 
        layer.Wo += Wo_updates
        layer.Wg += Wg_updates
        
        layer.bf += bf_updates 
        layer.bi += bi_updates 
        layer.bo += bo_updates
        layer.bg += bg_updates
        
    def post_update_params(self):
        self.iterations += 1
