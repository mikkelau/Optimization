# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:27:17 2024

@author: mikke
"""

from optimizer_GA import GeneticAlgorithmOptimizer 

from BeanFunction import function, upper_bounds, lower_bounds, gradients, hessian, f_opt, f_opt_tol
# from Brachistochrone import function, upper_bounds, lower_bounds
# from Rosenbrock import function, upper_bounds, lower_bounds, gradients, hessian
# from GoldsteinPrice import function, upper_bounds, lower_bounds, gradients, hessian # has local minima
# from TwoSpring import function, upper_bounds, lower_bounds, gradients, hessian
# from Rosenbrock_Stretched import function, upper_bounds, lower_bounds, gradients, hessian
# from BoothFunction import function, upper_bounds, lower_bounds, gradients, hessian
# from BukinFunction import function, upper_bounds, lower_bounds, gradients, hessian
# from EasomFunction import function, upper_bounds, lower_bounds, gradients, hessian
# from RsquaredPrimes import function
# from Ex5pt10 import function, upper_bounds, lower_bounds, gradients, hessian


# seed_num = 1
max_iters = 50
guess_range = [upper_bounds[0]-lower_bounds[0],upper_bounds[1]-lower_bounds[1]]
nVar = len(guess_range)
# seed(seed_num)
iteration_list = []
function_calls_list = []
num_solved = 0
numRuns = 1

for runNum in range(numRuns):
    # initial guess

    # initialize the optimizer
    optimizer = GeneticAlgorithmOptimizer(function, upper_bounds, lower_bounds, max_iters, plot_generations=True) # epsilon should be really small (zero) if the function isn't multimodal
    
    # call optimize
    optimizer.optimize()
    
    # print out important values
    optimizer.final_printout()
    
    # make a convergence plot
    optimizer.convergence_plot()
    
    # make a contour plot
    fig = optimizer.contour_plot(optimizer.x_list)
