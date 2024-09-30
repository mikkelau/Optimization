# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:48:35 2022

@author: mikke
"""

from optimizer_linesearch import LineSearchOptimizer 
from numpy import array
from random import random, seed
import numpy as np

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
# from JonesFunction import function, lower_bounds, upper_bounds, gradients, hessian
# from HartmannFunction import function, lower_bounds, upper_bounds, gradients, hessian

# from SteepestDescent import method
# from ConjugateGradient import method # should be used with BracketPinpoint or GoldenSectionSearch
# from NewtonsMethod import method
from BFGS import method

# from Backtrack import linesearch
# from BracketPinpoint import linesearch
from GoldenSectionSearch import linesearch
# from NewtonsMethod import linesearch # this just accepts the step as-is

# seed_num = 4
max_iters = 300
guess_range = np.array(upper_bounds)-np.array(lower_bounds)
nVar = len(guess_range)
# seed(seed_num)
iteration_list = []
function_calls_list = []
num_solved = 0
numRuns = 1


for runNum in range(numRuns):
    # initial guess
    # guess = array([(random()-0.5)*guess_range[i]+(upper_bounds[i]+lower_bounds[i])/2 for i in range(nVar)])
    # guess = array([0.54952189,0.91612188]) 
    # guess = array([-1,2]) # bean
    guess = array([9,-1]) # twospring
    # print('initial guess:',guess)

    # initialize the optimizer
    function.counter = 0
    linesearch_optimizer = LineSearchOptimizer(function, upper_bounds, lower_bounds, max_iters, min_step=np.finfo(np.float32).eps**(1/1))
    
    # call optimize
    linesearch_optimizer.optimize(method, linesearch, guess, gradients=gradients, hessian=hessian)
    # linesearch_optimizer.optimize(method, linesearch, guess)
    
    # print out important values
    linesearch_optimizer.final_printout()
    
    # make a contour plot
    linesearch_optimizer.contour_plot()
        
    # make a convergence plot
    linesearch_optimizer.convergence_plot()
    
    # plot objective function history
    linesearch_optimizer.obj_function_plot()
    
    
    # if (abs(linesearch_optimizer.function_value-f_opt) <= f_opt_tol):
    #     iteration_list.append(linesearch_optimizer.iterations)
    #     function_calls_list.append(linesearch_optimizer.function_calls)
    #     num_solved += 1


# print('seed:', seed_num)
# print('Average iterations:', sum(iteration_list)/num_solved)
# print('Average function calls:', sum(function_calls_list)/num_solved)
# print('Solved:',num_solved)
# print("\n")
