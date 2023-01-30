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
        linesearch.g = g
        method.iters = 0
        alpha = 1

        bounds_enforced = False
        while ((norm(g) > 1e-6) and (method.iters < max_iters)):
            
            # choose a search direction. should pass out a search direction and initial guess for alpha
            p, alpha = method(g, x, alpha, hessian, function, gradients) # pass in H or hessian?
            
            # linesearch
            f, g, alpha = linesearch(f, function, g, gradients, x, p, alpha)
                
            # enforce bounds (should I do this in the linesearch itself? 
            # No, the algorithm might get stuck because the bounds enforcement may make the point not good enough forever.
            # However, if the function is not defined outside the bounds then I'll run into issues)        
            alpha_new = alpha
            for i in range(len(x)):
                if (x[i]+alpha*p[i] > upper_bounds[i]):
                    # solve for the alpha that would land on the boundary
                    alpha_new = (upper_bounds[i]-x[i])/p[i]
                    if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                        alpha = alpha_new
                        bounds_enforced = True
                elif (x[i]+alpha*p[i] < lower_bounds[i]):
                    # solve for the alpha that would land on the boundary
                    alpha_new = (lower_bounds[i]-x[i])/p[i]
                    if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                        alpha = alpha_new
                        bounds_enforced = True
            # check for situations where the current x is on the boundary, and the proposed step will be outside the boundary, 
            # which would correct alpha to 0 and and remain in the same spot
            if ((alpha == 0.0) and (bounds_enforced == True)):
                print('method got stuck on boundary')
                break
                    
               
            # update x
            x = x+[alpha*i for i in p]
                    
            if bounds_enforced == True: # update f,g
               f = function(x)
               g = gradients(x,function) 
               bounds_enforced = False
            
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