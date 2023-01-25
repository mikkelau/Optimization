# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:27:45 2023

@author: mikke
"""

import FiniteDifference
from math import sqrt

upper_bounds = [-5,3]
lower_bounds = [-15,-3]

def function(X):
    x = X[0]
    y = X[1]
    
    f = 100*sqrt(abs(y-0.01*(x**2)))+0.01*abs(x+10)
    function.counter += 1
    return f

def gradients(X, function):
    
    g = FiniteDifference.gradients(X, function)
    
    return g

def hessian(X, function, gradients):
    
    H = FiniteDifference.hessian(X, function, gradients)
    
    return H