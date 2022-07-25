# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:32:09 2022

@author: mikke
"""
# bounds: -3 < x < 3; -1 < y < 3
def BeanFunction(X):
    x = X[0]
    y = X[1]

    f = (1-x)**2+(1-y)**2+0.5*(2*y-x**2)**2
    
    return f

def BeanFunctionGradients(X):
    x = X[0]
    y = X[1]
    
    g = [-2*(1-x)-2*x*(2*y-x**2), -2*(1-y)+2*(2*y-x**2)]
    
    return g