# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:48:35 2022

@author: mikke
"""

from numpy.linalg import norm
from random import random, seed
from numpy import array
from MakeContourPlot import MakeContourPlot
import matplotlib.pyplot as plt

# from RsquaredPrimes import function
# from BeanFunction import function, upper_bounds, lower_bounds, gradients, hessian
#from Rosenbrock import function, upper_bounds, lower_bounds
#from GoldsteinPrice import function, upper_bounds, lower_bounds
from TwoSpring import function, upper_bounds, lower_bounds

from FiniteDifference import gradients, hessian

# from SteepestDescent import method
# from ConjugateGradient import method
# from NewtonsMethod import method
from BFGS import method

# from Backtrack import linesearch
from BracketPinpoint import linesearch
# from NewtonsMethod import linesearch # this just accepts the step as-is


# set up, initialize
max_iters = 500
guess_range = [upper_bounds[0]-lower_bounds[0],upper_bounds[1]-lower_bounds[1]]
nVar = len(guess_range)
function.counter = 0
x_list = []
y_list = []

# initial guess
guess = array([(random()-0.5)*guess_range[i]+(upper_bounds[i]+lower_bounds[i])/nVar for i in range(nVar)])
guess = array([9, -1])
print('initial guess:',guess)
x = guess
x_list.append(x[0])
y_list.append(x[1])
g = gradients(x,function)
linesearch.g = g
method.iters = 0
method.g_old = g
f = function(x)
alpha = 1

bounds_enforced = False
while ((norm(g) > 1e-6) and (method.iters < max_iters)):

    # choose a search direction. should pass out a search direction and initial guess for alpha
    p, alpha = method(g, x, alpha, hessian, function, gradients) # pass in H or hessian?
    
    # linesearch
    f, g, alpha = linesearch(f, function, g, gradients, x, p, alpha)
        
    # enforce bounds (should I do this in the linesearch itself? No, the algorithm might get stuck because the bounds enforcement may make the point not good enough forever)        
    alpha_new = alpha
    for i in range(len(x)):
        if (x[i]+alpha*p[i] > upper_bounds[i]):
            # solve for the alpha that would land on the boundary
            alpha_new = (upper_bounds[i]-x[i])/p[i]
            if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                alpha = alpha_new
                bounds_enforced = True
        elif (x[i]+alpha*p[i] < lower_bounds[i]):
            # solve for the alpha that would land on the boundary
            alpha_new = (lower_bounds[i]-x[i])/p[i]
            if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                alpha = alpha_new
                bounds_enforced = True
    # check for situations where the current x is on the boundary, and the proposed step will be outside the boundary, 
    # which would correct alpha to 0 and and remain in the same spot
    if ((alpha == 0.0) and (bounds_enforced == True)):
        print('method got stuck')
        break
            
       
    # update x
    x = x+[alpha*i for i in p]
            
    if bounds_enforced == True: # update f,g
       f = function(x)
       g = gradients(x,function) 
       bounds_enforced = False
    
    # store the updated point
    x_list.append(x[0])
    y_list.append(x[1])
    

print("iterations:", method.iters)
print("function calls:", function.counter)
print("solution:",x)
print("function value:",f)    

MakeContourPlot(function,upper_bounds,lower_bounds)
plt.plot(x_list,y_list,c='red',marker='o',markerfacecolor='none')