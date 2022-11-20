# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:14:58 2022

@author: mikke
"""
upper_bounds = [1.79,3]
lower_bounds = [-1.79,-1]
def function(x):
    S = 0
    for i in range(len(x)-1):
        S += (100*((x[i+1]-x[i]**2)**2)+(1-x[i])**2)
   
    function.counter += 1
    return S

# problem guesses
# [1.94819887 2.1308015 ]
# [1.81722983 2.54906042]
# [-1.83990585 -0.04146654]
# [1.95263399 0.68405435]
# [-1.84151915 -0.95797539]
