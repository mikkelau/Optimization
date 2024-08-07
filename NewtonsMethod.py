# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:54:31 2022

@author: mikke
"""

from LU_factor import LU_factor
from optimizer_linesearch import LineSearchOptimizer
import os

def method(g, x, alpha, hessian, function, gradients): # g is a list, not an array

    if not hasattr(method, 'name'):
        method.name = os.path.basename(__file__).split('.')[0]

    H = hessian(x, function, gradients)
    
    p = -1*LU_factor(g, H) # -1 to go DOWNhill
        
    alpha = 1.0 # Newton step
    
    method.iters += 1
    
    return p, alpha

def linesearch(f, function, g, gradients, x, p, alpha, upper_bounds, lower_bounds, min_step, method_name):
    alpha = 1.0
    
    # enforce bounds
    alpha, Xnew, bounds_enforced = LineSearchOptimizer.enforce_bounds(alpha, x, p, upper_bounds, lower_bounds)
    
    # update x
    f_eval = function(Xnew)
    g_eval = gradients(Xnew,function)
    
    return f_eval, g_eval, alpha, Xnew