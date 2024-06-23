# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:44:19 2024

@author: mikke
"""

from optimizer_PSO import ParticleSwarmOptimizer 

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
# seed(seed_num)
numRuns = 1

for runNum in range(numRuns):
    # initial guess

    # initialize the optimizer
    optimizer = ParticleSwarmOptimizer(function, upper_bounds, lower_bounds, max_iters, plot_particles=True, num_pops=33) 
    
    # call optimize
    optimizer.optimize()
    
    # print out important values
    optimizer.final_printout()
    
    # make a convergence plot
    optimizer.convergence_plot()
    
    # make a contour plot
    fig = optimizer.contour_plot(optimizer.x_list)
