# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:42 2022

@author: mikke
"""
import numpy as np
from optimizer_linesearch import LineSearchOptimizer
from numpy.linalg import norm

def linesearch(f_current, function, g, gradients, X, p_dir, alpha, upper_bounds, lower_bounds, min_step, method_name): 
    # g is the gradient at current X
    # I pass in f_current to avoid another function eval
    
    #print('alpha:', alpha)
    
    mu = 1e-4
    rho = 0.5
    
    # enforce bounds      
    alpha, Xnew, bounds_enforced = LineSearchOptimizer.enforce_bounds(alpha, X, p_dir, upper_bounds, lower_bounds)
                
    # if alpha is less than the minimum step, define it as 0
    if ((norm(alpha*p_dir) < min_step) and (bounds_enforced == True)): 
        alpha = 0.0
        f_eval = f_current
        g_eval = g
        Xnew = X
    else:
        minimum_step_enforced = False
        dir_slope = np.dot(g, p_dir)  # this is the dot product
        f_eval = function(Xnew)
        entered_loop = False
        while (f_eval > f_current+mu*alpha*dir_slope):
            entered_loop = True
            
            alpha_new = rho*alpha
            # print('new alpha:',alpha_new)
            
            # Enforce the minimum step
            if (norm(alpha_new*p_dir) < min_step): # minimum step
                # print('enforcing minimum step')
                alpha_new = min_step/norm(p_dir)
                minimum_step_enforced = True

            # update quantities
            alpha = alpha_new    
            Xnew = X+alpha*p_dir
            f_eval = function(Xnew)
            
            if minimum_step_enforced:
                break
            
        else: # try increasing the step size if it was not shrunk
            if not (entered_loop or bounds_enforced) and (method_name in {"SteepestDescent","ConjugateGradient"}):
                                
                alpha_new = alpha/rho
                
                # enforce bounds      
                alpha_new, Xnew, bounds_enforced = LineSearchOptimizer.enforce_bounds(alpha_new, X, p_dir, upper_bounds, lower_bounds)
                
                f_new = function(X+alpha_new*p_dir)
                
                if not (f_new > f_current+mu*alpha_new*dir_slope): # if it still satisfies sufficient decrease 
                    # update quantities                               
                    alpha = alpha_new    
                    Xnew = X+alpha*p_dir
                    f_eval = f_new
            
        g_eval = gradients(Xnew, function)
    
    return f_eval, g_eval, alpha, Xnew