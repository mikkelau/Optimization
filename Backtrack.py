# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:42 2022

@author: mikke
"""
import numpy as np
from optimizer_linesearch import LineSearchOptimizer

def linesearch(f_current, function, g, gradients, X, p_dir, alpha, upper_bounds, lower_bounds): 
    # g is the gradient at current X
    # I pass in f_current to avoid another function eval
    
    #print('alpha:', alpha)
    
    mu = 1e-4
    rho = 0.5
    
    # enforce bounds      
    alpha, bounds_enforced = LineSearchOptimizer.enforce_bounds(alpha, X, p_dir, upper_bounds, lower_bounds)
                
    # if alpha is less than the minimum step, define it as 0
    if ((alpha < (np.finfo(np.float32).eps)**(1/3)) and (bounds_enforced == True)): # minimum step defined as epsilon^0.33
        alpha = 0.0
        f_eval = f_current
        g_eval = g
    else:
        minimum_step_enforced = False
        Xnew = X+[alpha*i for i in p_dir]
        dir_slope = np.dot(g, p_dir)  # this is the dot product
        f_eval = function(Xnew)
        while (f_eval > f_current+mu*alpha*dir_slope):
            alpha_new = rho*alpha
            # print('new alpha:',alpha_new)
            
            # Enforce the minimum step
            if (alpha_new < (np.finfo(np.float32).eps)**(1/3)): # minimum step defined as epsilon^0.33
                # print('enforcing minimum step')
                alpha_new = (np.finfo(np.float32).eps)**(1/3)
                minimum_step_enforced = True

            # update quantities
            alpha = alpha_new    
            Xnew = X+[alpha*i for i in p_dir]
            f_eval = function(Xnew)
            
            if minimum_step_enforced:
                break
            
            
        g_eval = gradients(Xnew, function)
    
    return f_eval, g_eval, alpha