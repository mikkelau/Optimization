# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:30:53 2022

@author: mikke
"""

def gradients(X, function):
    g = []
    delta = 0.000001
    for idx in range(len(X)):
        X_up = [i for i in X]
        X_dn = [i for i in X]
        X_up[idx] += delta
        X_dn[idx] -= delta
        g.append((function(X_up)-function(X_dn))/(2*delta))
    
    return g

def hessian(X, function, gradients):
    H = [[] for i in X]
    delta = 0.000001
    for idx in range(len(X)):
        X_up = [i for i in X]
        X_dn = [i for i in X]
        X_up[idx] += delta
        X_dn[idx] -= delta
        g_up = gradients(X_up, function)
        g_dn = gradients(X_dn, function)
        
        for i in range(len(H)):
            H[i].append((g_up[i]-g_dn[i])/(2*delta))
    
    return H