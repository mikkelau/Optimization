# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:48:35 2022

@author: mikke
"""

from optimizer_linesearch import LineSearchOptimizer 
from numpy import array
from random import random, seed

# from RsquaredPrimes import function
# from BeanFunction import function, upper_bounds, lower_bounds, gradients, hessian
from Rosenbrock import function, upper_bounds, lower_bounds, gradients, hessian
# from GoldsteinPrice import function, upper_bounds, lower_bounds, gradients, hessian
# from TwoSpring import function, upper_bounds, lower_bounds, gradients, hessian
# from Rosenbrock_Stretched import function, upper_bounds, lower_bounds, gradients, hessian

# from SteepestDescent import method
# from ConjugateGradient import method
from NewtonsMethod import method
# from BFGS import method

from Backtrack import linesearch
# from BracketPinpoint import linesearch
# from NewtonsMethod import linesearch # this just accepts the step as-is

max_iters = 500
guess_range = [upper_bounds[0]-lower_bounds[0],upper_bounds[1]-lower_bounds[1]]
nVar = len(guess_range)
# initial guess
guess = array([(random()-0.5)*guess_range[i]+(upper_bounds[i]+lower_bounds[i])/nVar for i in range(nVar)])

# initialize the optimizer
linesearch_optimizer = LineSearchOptimizer(function, upper_bounds, lower_bounds, max_iters, guess)

# call optimize
linesearch_optimizer.optimize(method, linesearch, gradients, hessian)

# print out important values
linesearch_optimizer.final_printout()

# make a convergence plot
linesearch_optimizer.convergence_plot()

# make a contour plot
linesearch_optimizer.contour_plot()



# print('seed:', seed_num)
# print('Average iterations:', sum(iterations)/len(iterations))
# print('average function calls:', sum(function_calls)/len(function_calls))
# print('did not solve:',did_not_solve)