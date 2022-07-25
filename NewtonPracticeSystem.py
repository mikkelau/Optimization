# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:13:54 2022

@author: mikke
"""

import math
from random import random
import matplotlib.pyplot as plt 
from numpy.linalg import norm
from numpy import array
from LU_factor import LU_factor

def mySystem(x):
    f = [[x[1]-1/x[0]], [x[1]-math.sqrt(x[0])]]
    g = [[1/(x[0]**2), 1], [-0.5*(1/math.sqrt(x[0])), 1]]
    
    return f,g
        
            
f = [[1], [1]]
upper_bounds = [30,30]
lower_bounds = [0.1,0.1]
guess_range = [upper_bounds[0]-lower_bounds[0],upper_bounds[1]-lower_bounds[1]]
nVar = len(upper_bounds)

# generate guess
# center points in the bounds of the domain
#guess = array([(random()-0.5)*guess_range[0]+(upper_bounds[0]+lower_bounds[0])/len(f), (random()-0.5)*guess_range[1]+(upper_bounds[0]+lower_bounds[0])/len(f)])
guess = [0.1, -100]

resids = []
print('starting guess:', guess)
while (norm(f) > 1e-14):
    f,G = mySystem(guess)
    delta_u = LU_factor(f, G)
    guess = guess-delta_u
    
    # enforce bounds
    for j in range(nVar):
        if guess[j] < lower_bounds[j]:
            guess[j] = lower_bounds[j]
        elif guess[j] > upper_bounds[j]:
            guess[j] = upper_bounds[j]
    
    print('residual:',norm(f))
    resids.append(norm(f))

print('solution:',guess)
plt.plot(resids)

# naming the x axis 
plt.xlabel('Iterations') 
# naming the y axis 
plt.ylabel('Residual') 

plt.yscale("log")
  
# function to show the plot 
plt.show() 