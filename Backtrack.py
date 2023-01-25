# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:42 2022

@author: mikke
"""
import numpy as np

def linesearch(f_current, function, g, gradients, X, p_dir, alpha): 
    # g is the gradient at current X
    # I pass in f_current to avoid another function eval
    #print('alpha:', alpha)
    
    mu = 1e-4
    rho = 0.5
    Xnew = X+[alpha*i for i in p_dir]
    dir_slope = sum([i*j for (i, j) in zip(g, p_dir)]) # this is the dot product
    f_eval = function(Xnew)
    while (f_eval > f_current+mu*alpha*dir_slope):
        alpha = rho*alpha
        # print('new alpha:',alpha)
        
        # Take the minimum step if needed. Do I want to implement a minimum step?
        if (alpha < (np.finfo(np.float32).eps)**(1/3)): # minimum step defined as epsilon^0.33
            print('enforcing minimum step')
            break

        # update Xnew
        Xnew = X+[alpha*i for i in p_dir]
        f_eval = function(Xnew)
        
    g_eval = gradients(Xnew, function)
    
    # update internal quantities
    linesearch.p_old = p_dir
    linesearch.alpha = alpha
    
    return f_eval, g_eval, alpha