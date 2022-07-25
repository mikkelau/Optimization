# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:48:35 2022

@author: mikke
"""
from numpy.linalg import norm
from random import random
from numpy import array
from BeanFunction import BeanFunction, BeanFunctionGradients
from SteepestDescent import SteepestDescent
from Backtrack import Backtrack
from MakeContourPlot import MakeContourPlot
import matplotlib.pyplot as plt

function = lambda x: BeanFunction(x)
gradients = lambda x: BeanFunctionGradients(x)
method = lambda g: SteepestDescent(g)
linesearch = lambda f_current, function, g, alpha, x, p_dir: Backtrack(f_current, function, g, alpha, x, p_dir)

# set up, initialize
max_iters = 100
upper_bounds = [3,3]
lower_bounds = [-3,-1]
guess_range = [upper_bounds[0]-lower_bounds[0],upper_bounds[1]-lower_bounds[1]]
nVar = len(guess_range)
MakeContourPlot(function,upper_bounds,lower_bounds)
x_list = []
y_list = []
p_old = [random() for i in range(nVar)]
g_old = [random() for i in range(nVar)]
g = [random()+1 for i in range(nVar)] # dummy gradients

# initial guess
guess = array([(random()-0.5)*guess_range[i]+(upper_bounds[i]+lower_bounds[i])/nVar for i in range(nVar)])
x = guess
x_list.append(x[0])
y_list.append(x[1])
g = gradients(x)
f = function(x)

iters = 0
while (norm(g) > 1e-6) and (iters < max_iters):
    
    # choose a search direction
    p = method(g)
    
    # linesearch
    if iters == 0:
        alpha = norm(guess_range)/5 # this is totally arbitrary, not sure what a good start is
    else:
        alpha = alpha*(sum(i*j for i,j in zip(g_old, p_old))/sum(i*j for i,j in zip(g, p)))
    # do the search
    x, f, alpha = linesearch(f, function, g, alpha, x, p)
        
    # store the updated point
    x_list.append(x[0])
    y_list.append(x[1])
    
    # keep track of previous values before updating them
    p_old = p
    g_old = g
    
    # update gradient at new location
    g = gradients(x)
    
    iters += 1

print("iterations:",iters)
print("solution:",x)
print("function value:",f)
plt.plot(x_list,y_list,c='red',marker='o',markerfacecolor='none')
