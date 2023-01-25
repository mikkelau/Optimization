# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:27:45 2023

@author: mikke
"""

import FiniteDifference
from math import pi, exp, cos

upper_bounds = [4.75,4.75]
lower_bounds = [1.5,1.5]

def function(X):
    x = X[0]
    y = X[1]
    
    f = -1*cos(x)*cos(y)*exp(-1*(((x-pi)**2)+((y-pi)**2)))
    function.counter += 1
    return f

def gradients(X, function):
    
    g = FiniteDifference.gradients(X, function)
    
    return g

def hessian(X, function, gradients):
    
    H = FiniteDifference.hessian(X, function, gradients)
    
    return H