# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:30:53 2022

@author: mikke
"""
import numpy as np

def gradients(X, function):
    g = np.empty((len(X),))
    delta = (np.finfo(np.float32).eps)**(1/3)
    for idx in range(len(X)):
        X_up = np.array([i for i in X],dtype='float64') # need to set type as float64 or else the += operation may not work correctly
        X_dn = np.array([i for i in X],dtype='float64') # need to set type as float64 or else the += operation may not work correctly
        X_up[idx] += delta
        X_dn[idx] -= delta

        g[idx] = (function(X_up)-function(X_dn))/(2*delta)
    
    return g

def hessian(X, function, gradients):
    H = np.empty((len(X),len(X)))
    delta = (np.finfo(np.float32).eps)**(1/3)
    for idx in range(len(X)):
        X_up = np.array([i for i in X],dtype='float64') # need to set type as float64 or else the += operation may not work correctly
        X_dn = np.array([i for i in X],dtype='float64') # need to set type as float64 or else the += operation may not work correctly
        X_up[idx] += delta
        X_dn[idx] -= delta
        g_up = gradients(X_up, function)
        g_dn = gradients(X_dn, function)
        
        for i in range(len(H)):
            H[i][idx] = (g_up[i]-g_dn[i])/(2*delta)
    
    return H