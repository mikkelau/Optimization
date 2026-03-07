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

class SimulatedAnnealingOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        
    def optimize(self, x0):
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        
        n = len(upper_bounds)
        
        # initialize some stuff
        f = function(x0)
        x = x0
        self.x_list.append(x)
        self.f_list.append(f)
                
        for k in range(max_iters):
            
            # set the temperature
            T = temperature(1-(k+1)/max_iters)
            
            # choose a neighboring solution
            
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
                
            
            