# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:06:35 2023

@author: mikke
"""
from numpy import array
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Optimizer:
    def __init__(self, function, upper_bounds, lower_bounds, max_iters):
        self.solution = array([])
        self.convergence = None
        self.iterations = 0
        self.function_calls = 0
        self.function_value = None
        self.function = function
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds 
        self.max_iters = max_iters 

    def final_printout(self):
        print("iterations:", self.iterations)
        print("function calls:", self.function_calls)
        print("solution:",self.solution)
        print("function value:",self.function_value)
        print("\n")
    
    def convergence_plot(self):
        plt.figure()
        plt.yscale("log")
        plt.plot(self.convergence)
        plt.grid()
        plt.xlabel('ITERATIONS',fontweight='bold')
        plt.ylabel('CONVERGENCE',fontweight='bold')
        
        # make the iterations axis only show integers
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))    