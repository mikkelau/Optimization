# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 09:48:19 2022

@author: mikke
"""
# limits are -5 <= x,y <= 5
def Himmelblau(X):
    x = X[0]
    y = X[1]

    f = (x**2+y-11)**2+(x+y**2-7)**2
    
    return f