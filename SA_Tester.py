# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:27:17 2024

@author: mikke
"""

from optimizer_SA import SimulatedAnnealingOptimizer 
from random import random, seed
import numpy as np
from numpy import array

# from BeanFunction import function, upper_bounds, lower_bounds, gradients, hessian, f_opt, f_opt_tol
# from Brachistochrone import function, upper_bounds, lower_bounds
# from Rosenbrock import function, upper_bounds, lower_bounds, gradients, hessian
# from GoldsteinPrice import function, upper_bounds, lower_bounds, gradients, hessian # has local minima
from TwoSpring import function, upper_bounds, lower_bounds, gradients, hessian
# from Rosenbrock_Stretched import function, upper_bounds, lower_bounds, gradients, hessian
# from BoothFunction import function, upper_bounds, lower_bounds, gradients, hessian
# from BukinFunction import function, upper_bounds, lower_bounds, gradients, hessian
# from EasomFunction import function, upper_bounds, lower_bounds, gradients, hessian
# from RsquaredPrimes import function
# from Ex5pt10 import function, upper_bounds, lower_bounds, gradients, hessian # tests boundary behavior


# seed_num = 1
max_iters = 1000
# seed(seed_num)
guess_range = np.array(upper_bounds)-np.array(lower_bounds)
nVar = len(guess_range)
numRuns = 1
temp = 10

for runNum in range(numRuns):
    # initial guess
    # guess = array([(random()-0.5)*guess_range[i]+(upper_bounds[i]+lower_bounds[i])/2 for i in range(nVar)])
    guess = array([9,-1]) # twospring
    
    # initialize the optimizer
    optimizer = SimulatedAnnealingOptimizer(function, upper_bounds, lower_bounds, max_iters, temp)
    
    # call optimize
    optimizer.optimize(guess)
    
    # print out important values
    optimizer.final_printout()
    
    # make a convergence plot
    optimizer.convergence_plot()
    
    # make a contour plot
    fig = optimizer.contour_plot(optimizer.x_list)
