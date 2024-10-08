# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:36:02 2023

@author: mikke
"""

from math import sqrt
import numpy as np

upper_bounds = [1.0, 1.0]
lower_bounds = [0.0, 0.0]


def function(Y):
    S = 0
    
    mu = 0.3
    H = 1.0
    L = 1.0
    
    # add endpoints to the Y position vector
    Y = np.array(Y)
    Y = np.insert(Y,0,H)
    Y = np.append(Y,0.0)
    
    length = len(Y)
        
    X = np.array([0.0 + i*(L-0.0)/(length-1) for i in range(length)])
    
    for idx in range(len(Y)-1):
        delx = X[idx+1]-X[idx]
        dely = Y[idx+1]-Y[idx]
        
        # Avoid taking the square root of a negative number
        S += sqrt((delx**2)+(dely**2))/(sqrt(max(H-Y[idx+1]-mu*X[idx+1],0.0))+sqrt(max(H-Y[idx]-mu*X[idx],0.0)))
    
    f = S
    
    if hasattr(function,'counter'):
        function.counter += 1

    return f
    
    