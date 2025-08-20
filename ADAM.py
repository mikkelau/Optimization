# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 13:56:55 2022

@author: mikke
"""

import numpy as np
import os
from optimizer_linesearch import LineSearchOptimizer

def method(g, x, alpha, hessian, function, gradients):  # g is a list, not an array

    if not hasattr(method, 'name'):
        method.name = os.path.basename(__file__).split('.')[0]
        
    # default settings
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    if (method.iters == 0):
        m_old = np.zeros(len(x))
        v_old = np.zeros(len(x))
    else:
        m_old = method.m_old
        v_old = method.v_old

    m = beta1*m_old+(1-beta1)*g
    v = beta2*v_old+(1-beta2)*(g**2)

    method.iters += 1
    
    # Apply bias correction
    m_hat = m/(1-beta1**method.iters)
    v_hat = v/(1-beta2**method.iters)
    
    p = -1*m_hat/(np.sqrt(v_hat)+eps)

    alpha = 1.0 # Newton step

    # store values for next iteration
    method.m_old = m
    method.v_old = v
    
    return p, alpha

def linesearch(f, function, g, gradients, x, p, alpha, upper_bounds, lower_bounds, min_step, method_name):
    alpha = 0.001
    
    # enforce bounds
    alpha, Xnew, bounds_enforced = LineSearchOptimizer.enforce_bounds(alpha, x, p, upper_bounds, lower_bounds)
    
    # update
    f_eval = function(Xnew)
    g_eval = gradients(Xnew,function)
    
    return f_eval, g_eval, alpha, Xnew