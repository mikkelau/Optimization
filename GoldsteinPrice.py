# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:05:22 2022

@author: mikke
"""
# limits are -2 <= x,y <= 2
upper_bounds = [2,2]
lower_bounds = [-2,-2]
def function(X):

    x = X[0]
    y = X[1]

    f = (1+((x+y+1)**2)*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*(30+((2*x-3*y)**2)*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    
    function.counter += 1
    
    return f