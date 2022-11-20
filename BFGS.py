# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 13:56:55 2022

@author: mikke
"""

import numpy as np
from LU_factor import LU_factor

def method(g, x, alpha, hessian, function, gradients):  # g is a list, not an array

    I = np.identity(len(x))
    
    if (method.iters == 0):
        H = I
        V = I
    else:
        H_old = method.H_old
        
        V_old = method.V_old
        g_old = method.g_old
        x_old = method.x_old
        
        y = np.array([i-j for (i,j) in zip(g,g_old)])
        s = np.array([i-j for (i,j) in zip(x,x_old)])
        
        H = H_old + np.outer(y,y)/np.dot(y,s)-np.dot(np.outer(H_old.dot(s),s),H_old)/np.dot(s.dot(H_old),s)
        
        sigma = 1/y.dot(s)
        V = (I-sigma*np.outer(s,y))*V_old*(I-sigma*np.outer(y,s))+sigma*np.outer(s,s)

    p = LU_factor(g, H)
        
    # p = np.dot(V,g)
    
    p = [i*-1 for i in p]
    
    alpha = 1.0 # Newton step
    
    method.H_old = H
    method.V_old = V
    method.g_old = g
    method.x_old = x
    method.iters += 1
    
    return p, alpha