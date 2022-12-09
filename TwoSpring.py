# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 18:52:00 2022

@author: mikke
"""
# Consider a mass supported by two springs, as shown
# in Fig. 4.55. Formulating the total potential energy for the system as a function
# of the mass position yields the following problem:
    
upper_bounds = [18,12]
lower_bounds = [-8,-10]
def function(X):   
    x = X[0]
    y = X[1]
    
    l1 = 12
    l2 = 8
    k1 = 1
    k2 = 10
    mg = 7

    f = 0.5*k1*((((((l1+x)**2)+(y**2))**0.5)-l1)**2)+0.5*k2*((((((l2-x)**2)+(y**2))**0.5)-l2)**2)-mg*y
    
    function.counter += 1
    
    return f