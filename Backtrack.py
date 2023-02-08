# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:42 2022

@author: mikke
"""
import numpy as np

def linesearch(f_current, function, g, gradients, X, p_dir, alpha, upper_bounds, lower_bounds): 
    # g is the gradient at current X
    # I pass in f_current to avoid another function eval
    #print('alpha:', alpha)
    
    mu = 1e-4
    rho = 0.5
    
    # enforce bounds      
    bounds_enforced = False
    alpha_new = alpha
    for i in range(len(X)):
        if (X[i]+alpha*p_dir[i] > upper_bounds[i]):
            # solve for the alpha that would land on the boundary
            alpha_new = (upper_bounds[i]-X[i])/p_dir[i]
            if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                alpha = alpha_new
                bounds_enforced = True
        elif (X[i]+alpha*p_dir[i] < lower_bounds[i]):
            # solve for the alpha that would land on the boundary
            alpha_new = (lower_bounds[i]-X[i])/p_dir[i]
            if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                alpha = alpha_new
                bounds_enforced = True
                
    # if alpha is less than the minimum step, define it as 0
    if ((alpha < (np.finfo(np.float32).eps)**(1/3)) and (bounds_enforced == True)): # minimum step defined as epsilon^0.33
        alpha = 0.0
        f_eval = f_current
        g_eval = g
    else:
        Xnew = X+[alpha*i for i in p_dir]
        dir_slope = sum([i*j for (i, j) in zip(g, p_dir)]) # this is the dot product
        f_eval = function(Xnew)
        while (f_eval > f_current+mu*alpha*dir_slope):
            alpha_new = rho*alpha
            # print('new alpha:',alpha_new)
            
            # Enforce the minimum step if needed. Do I want to implement a minimum step?
            if (alpha_new < (np.finfo(np.float32).eps)**(1/3)): # minimum step defined as epsilon^0.33
                # print('enforcing minimum step')
                # breaking here just accepts the previous alpha and other quantities. May want to actually set alpha to be the minimum step and update everything
                break
            else:
                # update quantities
                alpha = alpha_new    
                Xnew = X+[alpha*i for i in p_dir]
                f_eval = function(Xnew)
            
        g_eval = gradients(Xnew, function)
    
    # update internal quantities. Are these needed?
    linesearch.p_old = p_dir
    linesearch.alpha = alpha
    
    return f_eval, g_eval, alpha