# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:32:09 2022

@author: mikke
"""
 
# bounds: -3 < x < 3; -1 < y < 3
upper_bounds = [2.5,3]
# upper_bounds = [1,3] # to test boundary behavior
lower_bounds = [-2.5,-1]
f_opt = 0.09194
f_opt_tol = 5e-5

def function(X):   
    x = X[0]
    y = X[1]

    f = (1-x)**2+(1-y)**2+0.5*(2*y-x**2)**2
    
    if hasattr(function,'counter'):
        function.counter += 1
    
    return f

def gradients(X, function):
    x = X[0]
    y = X[1]
    
    g = [-2*(1-x)-2*x*(2*y-x**2), -2*(1-y)+2*(2*y-x**2)]
    
    return g

def hessian(X, function, gradients):
    x = X[0]
    y = X[1]
    
    H = [[2-4*y+6*x**2, -4*x], [-4*x, 6]]
    
    return H
