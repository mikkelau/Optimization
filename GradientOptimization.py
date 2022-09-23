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

from RsquaredPrimes import function
# from BeanFunction import function
# from BeanFunction import gradients
# from BeanFunction import hessian

from FiniteDifference import gradients
from FiniteDifference import hessian

#from SteepestDescent import method
#from ConjugateGradient import method
from NewtonsMethod import method

#from Backtrack import linesearch
from BracketPinpoint import linesearch
#from NewtonsMethod import linesearch # this just accepts the Newton step as-is


# set up, initialize
max_iters = 500
# might be a good idea to include poinds in the function, like function.upper_bounds, etc.
upper_bounds = [11,-1]
lower_bounds = [3,-200]
guess_range = [upper_bounds[0]-lower_bounds[0],upper_bounds[1]-lower_bounds[1]]
nVar = len(guess_range)
function.counter = 0
x_list = []
y_list = []

# initial guess
guess = array([(random()-0.5)*guess_range[i]+(upper_bounds[i]+lower_bounds[i])/nVar for i in range(nVar)])
#guess = array([-0.40903151,  0.69488563]) # newton gets stuck here
x = guess
x_list.append(x[0])
y_list.append(x[1])
g = gradients(x,function)
linesearch.g = g
method.iters = 0
method.g_old = g
f = function(x)
alpha = 1

iters = 0
while ((norm(g) > 1e-6) and (iters < max_iters)):
    
    # choose a search direction. should pass out a search direction and initial guess for alpha
    p, alpha = method(g, x, alpha, hessian, function, gradients) # pass in H or hessian?
    
    # linesearch
    f, g, alpha = linesearch(f, function, g, gradients, x, p, alpha)
        
    # update x
    #x = [j+k for (j,k) in zip(list(x),[alpha*i for i in p])]
    x = x+[alpha*i for i in p]
    
    # enforce bounds (should I do this in the linesearch itself? No, the algorithm might get stuck)
    for i in range(len(x)):
        if (x[i] > upper_bounds[i]):
            x[i] = upper_bounds[i]
        elif (x[i] < lower_bounds[i]):
            x[i] = lower_bounds[i]
    
    # store the updated point
    x_list.append(x[0])
    y_list.append(x[1])
    
    iters += 1
    #print('iter:',iters)

print("iterations:",iters)
print("function calls:", function.counter)
print("solution:",x)
print("function value:",f)    

MakeContourPlot(function,upper_bounds,lower_bounds)
plt.plot(x_list,y_list,c='red',marker='o',markerfacecolor='none')
