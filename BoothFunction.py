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
    
    f = ((x+2*y-7)**2)+((2*x+y-5)**2)
    function.counter += 1
    return f

def gradients(X, function):
    
    g = FiniteDifference.gradients(X, function)
    
    return g

def hessian(X, function, gradients):
    
    H = FiniteDifference.hessian(X, function, gradients)
    
    return H