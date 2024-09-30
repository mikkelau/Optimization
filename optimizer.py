# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:06:35 2023

@author: mikke
"""
from numpy import array
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import copy
import numpy as np
from matplotlib import ticker

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
        fig = plt.figure()
        plt.yscale("log")
        convergence = np.array(self.convergence,dtype=float)
        if convergence[-1] <= 0:
            convergence = (convergence-convergence[-1]+1.0)
        plt.plot(convergence)
        plt.grid()
        plt.xlabel('ITERATIONS',fontweight='bold')
        plt.ylabel('CONVERGENCE',fontweight='bold')
        
        # make the iterations axis only show integers
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   
        
        return fig
        
    def make_contour_plot(self,function,upper_lims,lower_lims):
        numpoints = 100
        feature_x = np.linspace(lower_lims[0], upper_lims[0], numpoints)
        feature_y = np.linspace(lower_lims[1], upper_lims[1], numpoints)
        
        # Creating 2-D grid of features
        [X, Y] = np.meshgrid(feature_x, feature_y)
        Z = np.empty((numpoints,numpoints))
        for col in range(numpoints):
            for row in range(numpoints):
                Z[row,col] = function((feature_x[col], feature_y[row])) # function inputs defined as a tuple, not sure if that matters
                
        # create filled contour plot
        fig, ax = plt.subplots(1, 1)
        if ((Z.max() > 1000*Z.min()) and (Z.min() > 0)): # check if a logarithmic contour plot is needed        
            cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator())
            # plot colorbar
            cbar = fig.colorbar(cs, format='%.0e')
        else:
            cs = ax.contourf(X, Y, Z, 16)
            # plot colorbar
            cbar = fig.colorbar(cs)
            
        # set the axes to be on the same scale
        ax.set_aspect('equal', adjustable='box')  
        
        cfm = plt.get_current_fig_manager()
        cfm.window.activateWindow()
        cfm.window.raise_()
            
        # ax.set_title('Filled Contour Plot')
        # ax.set_xlabel('feature_x')
        # ax.set_ylabel('feature_y')
        
        return fig