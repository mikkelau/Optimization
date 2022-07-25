# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 20:27:56 2022

@author: mikke
"""

def PracticeFunction(X):
    x = X[0]
    y = X[1]

    f = (1-x)**2+(1-y)**2+0.5*(2*y-x**2)**2
    
    return f