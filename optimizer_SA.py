# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 05:19:54 2026

@author: mikke
"""

import optimizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import random
import time
import math
from numpy.random import randn

class SimulatedAnnealingOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, T0, neighbor=None):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        self.T0 = T0
        if neighbor:
            self.neighbor = neighbor
        
    def contour_plot(self,points=None):
        if len(self.upper_bounds) == 2:
            if points is None:
                points = self.x_list
            # enable interactive mode
            plt.ion()
            fig = self.make_contour_plot(self.function, self.upper_bounds, self.lower_bounds)
            # plot the points that got passed in
            plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none')
            return fig
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n")
        
    def temperature(self, T0, itr,  max_iters):
        # # exponential decrease
        # T = T0*(0.99**itr)
        
        T = T0*(float(1-itr/max_iters)**2)
                
        # # fast annealing
        # T = T0/float(itr+1)
        
        return T
    
    def neighbor(self, x):
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        step_size = (np.array(upper_bounds)-np.array(lower_bounds))/100
        n = len(upper_bounds)
        
        xnew = x+randn(n)*step_size
        
        return xnew
        
    def optimize(self, x0):
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        T0 = self.T0
        neighbor = self.neighbor
                
        # initialize some stuff
        function.counter = 0
        f = function(x0)
        x = x0
        self.x_list.append(x)
        self.f_list.append(f)
        
                
        for k in range(max_iters):
            # set the temperature
            T = self.temperature(T0,k,max_iters)
            
            # choose a neighboring solution
            xnew = neighbor(self,x)
            xnew = np.clip(xnew, lower_bounds, upper_bounds) # enforce bounds, not sure if this should be handled inside the neighbor function
            
            fnew = function(xnew)
            
            # determine probability of accepting new solution
            P = min(math.exp((f-fnew)/T),1)
            
            # randomly draw from uniform distribution
            r = random.random()
            
            if P >= r:
                x = xnew
                f = fnew
                
            # store the updated point and associated fitness value
            self.x_list.append(x)
            self.f_list.append(f)
                
        self.iterations = k+1
        self.function_calls = function.counter
        self.solution = x
        self.function_value = f
        self.convergence = self.f_list            
            