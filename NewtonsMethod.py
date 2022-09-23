# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:54:31 2022

@author: mikke
"""

from LU_factor import LU_factor

def method(g, x, alpha, hessian, function, gradients): # g is a list, not an array

    H = hessian(x, function, gradients)
    
    p = LU_factor(g, H)
    
    p = [i*-1 for i in p]
    
    alpha = 1.0 # Newton step
    
    return p, alpha

def linesearch(f, function, g, gradients, x, p, alpha):
    alpha = 1
    
    # update x
    Xnew = x+[alpha*i for i in p]
    f_eval = function(Xnew)
    g_eval = gradients(Xnew,function)
    
    return f_eval, g_eval, alpha