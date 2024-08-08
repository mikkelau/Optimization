# -*- coding: utf-8 -*-
"""
Created on Sat May 25 13:21:59 2024

@author: mikke
"""

""" 
A Generalized Pattern Search (GPS) implementation.

Possible improvements:
-No need to evaluate all LHS points in the search method. Could break as soon as any point is found that is better than the current point.
-Search function could randomly evaluate some number of points from an initial grid of points with very small spacing.
"""

import optimizer
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import qmc

class GPSOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, tol=1e-6):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.guess = []
        self.x_list = []
        self.f_list = []
        self.tol = tol
        
    def contour_plot(self,points):
        if len(self.guess) == 2:
            # enable interactive mode
            plt.ion()
            fig = self.make_contour_plot(self.function, self.upper_bounds, self.lower_bounds)
            # plot the points that got passed in
            plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none')
            return fig
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n")  
    
    def search(self, x, f, point_to_value):
        success = False
        xnew = x
        fnew = f
        
        # create a LHS
        sampler = qmc.LatinHypercube(d=len(x))
        sample = sampler.random(n=10)
        points = qmc.scale(sample, self.lower_bounds, self.upper_bounds)
        
        # Order points from the lowest (best) to the highest
        sample_dict = {}
        for point in points:
            if tuple(point) not in point_to_value:
                point_to_value[tuple(point)] = self.function(point)
            sample_dict[tuple(point)] = point_to_value[tuple(point)]
        sample_dict = OrderedDict(sorted(sample_dict.items(), key=lambda item: item[1]))
        
        if list(sample_dict.values())[0] < f:
            xnew = list(sample_dict.keys())[0]
            fnew = list(sample_dict.values())[0]
            success = True            
        
        return success, xnew, fnew, point_to_value
            
    def optimize(self, x0):
        self.guess = x0
        
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        f_list = []
        
        function.counter = 0
        iters = 0
        
        # enforce bounds in initial guess
        x0_new = np.clip(x0, lower_bounds, upper_bounds)

        if not np.array_equal(x0_new, x0):
            print('Bounds enforced for initial guess')
        x0 = x0_new
        
        n = len(x0)
        
        # define the spanning set D
        D = np.vstack([np.identity(n),-1*np.identity(n)])
        
        # create a dictionary to store points and their corresponding function values
        point_to_value = {}
        
        # populate the dictionary with the initial guess
        point_to_value[tuple(x0)] = function(x0)
        
        # define delta
        ranges = np.array([i-j for i,j in zip(upper_bounds,lower_bounds)])
        delta = 0.25*min(ranges) # this is arbitrary, it might be good to determine this using some other criteria
        delta_max = 0.65*min(ranges) # this is arbitrary, it might be good to determine this using some other criteria
        
        while ((delta > self.tol) and (iters < max_iters)):
            self.x_list.append(x0)
            f_list.append(point_to_value[tuple(x0)])
            
            search_success, x0, f, point_to_value = self.search(x0, point_to_value[tuple(x0)], point_to_value)
            if search_success:
                delta = min(2*delta,delta_max)
                iters += 1
                continue
            else:
                poll_success = False
                for i in range(n*2):
                    
                    s = x0 + delta*D[i]
                    
                    # enforce bounds
                    s = np.clip(s, lower_bounds, upper_bounds)
                    
                    # check if the new point has already been evaluated
                    if tuple(s) not in point_to_value:
                        point_to_value[tuple(s)] = function(s)
                        
                    # opportunistic polling
                    if point_to_value[tuple(s)] < f:
                        x0 = s
                        f = point_to_value[tuple(s)]
                        poll_success = True
                        break
                    
            if not poll_success:
                delta *= 0.5
                
            iters += 1
            
        f_list.append(point_to_value[tuple(x0)])
        self.x_list.append(x0)
        self.iterations = iters
        self.function_calls = function.counter
        self.solution = x0
        self.function_value = point_to_value[tuple(x0)]
        self.convergence = f_list