# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:54:31 2022

@author: mikke
"""

from numpy.linalg import norm
import numpy as np

def method(g, x, alpha, hessian, function, gradients): # g is a list, not an array
    
    if (method.iters == 0) or (abs(np.dot(g, method.g_old)/np.dot(g, g)) >= 0.1):
        p = np.array([-1*i/norm(g) for i in g])
    else:
        print('utilized beta')
        # Fletcher–Reeves formula
        # beta = sum([i*j for (i, j) in zip(g, g)])/sum([i*j for (i, j) in zip(method.g_old, method.g_old)]) 
        
        # Polak–Ribière formula
        beta = np.dot(g, g-method.g_old)/np.dot(method.g_old, method.g_old) 
        
        beta = max(0.0,beta) # force Beta to not be negative
        p = np.array([-1*i/norm(g) for i in g])+beta*method.p_old
        
    if method.iters == 0:
        alpha = 1 # this is totally arbitrary, not sure what a good size is
    else:
        alpha = alpha*abs((np.dot(method.g_old, method.p_old)/np.dot(g, p)))

    # update internal quantities
    method.p_old = p
    method.g_old = g
    method.iters += 1
    
    return p, alpha