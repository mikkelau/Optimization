# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:42 2022

@author: mikke
"""

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
     #   print('new alpha:',alpha)

        # update Xnew
        Xnew = X+[alpha*i for i in p_dir]
        f_eval = function(Xnew)
        
    g_eval = gradients(Xnew, function)
    
    # update internal quantities
    linesearch.p_old = p_dir
    linesearch.alpha = alpha
    
    return f_eval, g_eval, alpha