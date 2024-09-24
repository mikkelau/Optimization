# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:27:45 2023

@author: mikke
"""

import FiniteDifference

upper_bounds = [10,10]
lower_bounds = [-10,-10]

def function(X):
    x = X[0]
    y = X[1]
    f = 0.26*((x**2)+(y**2))-0.48*x*y   
    
    if hasattr(function,'counter'):
        function.counter += 1
        
    return f

def gradients(X, function):
    
    g = FiniteDifference.gradients(X, function)
    
    return g

def hessian(X, function, gradients):
    
    H = FiniteDifference.hessian(X, function, gradients)
    
    return H