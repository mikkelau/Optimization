# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:32:14 2023

@author: mikke
"""

import optimizer_gradients
from numpy.linalg import norm
import FiniteDifference

class LineSearchOptimizer(optimizer_gradients.GradientBasedOptimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, x0):
        super().__init__(function, upper_bounds, lower_bounds, max_iters, x0)
    
    def optimize(self, method, linesearch, gradients=FiniteDifference.gradients, hessian=FiniteDifference.hessian):
        x = self.guess
        if (len(x)==2):
            self.x_list = []
            self.y_list = []
            self.x_list.append(x[0])
            self.y_list.append(x[1])
        g_list = []
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        
        function.counter = 0
        f = function(x)
        g = gradients(x,function)
        g_list.append(norm(g))
        method.iters = 0
        alpha = 1

        while ((norm(g) > 1e-6) and (method.iters < max_iters)):
            
            # choose a search direction. should pass out a search direction and initial guess for alpha
            p, alpha = method(g, x, alpha, hessian, function, gradients) # pass in H or hessian?
            
            # linesearch
            f, g, alpha = linesearch(f, function, g, gradients, x, p, alpha, upper_bounds, lower_bounds)
                
            # check if alpha was forced to 0 (due to boundary enforcement)
            if (alpha == 0.0):
                print('method got stuck on boundary')
                break   
               
            # update x
            x = x+[alpha*i for i in p]
            
            # store the updated point and associated gradient
            if (len(x)==2):
                self.x_list.append(x[0])
                self.y_list.append(x[1])
            g_list.append(norm(g))
            
        self.iterations = method.iters
        self.function_calls = function.counter
        self.solution = x
        self.function_value = f
        self.convergence = g_list