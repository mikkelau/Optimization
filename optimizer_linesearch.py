# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:32:14 2023

@author: mikke
"""

import optimizer_gradients
from numpy.linalg import norm
import FiniteDifference
import numpy as np

class LineSearchOptimizer(optimizer_gradients.GradientBasedOptimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        
    def enforce_bounds(alpha, X, p_dir, upper_bounds, lower_bounds):
        bounds_enforced = False
        for i in range(len(X)):
            if (X[i]+alpha*p_dir[i] > upper_bounds[i]):
                # solve for the alpha that would land on the boundary
                alpha_new = (upper_bounds[i]-X[i])/p_dir[i]
                if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                    alpha = alpha_new
                    bounds_enforced = True
            elif (X[i]+alpha*p_dir[i] < lower_bounds[i]):
                # solve for the alpha that would land on the boundary
                alpha_new = (lower_bounds[i]-X[i])/p_dir[i]
                if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                    alpha = alpha_new
                    bounds_enforced = True
        
        return alpha, bounds_enforced
    
    def optimize(self, method, linesearch, x0, gradients=FiniteDifference.gradients, hessian=FiniteDifference.hessian):
        self.guess = x0
        x = x0
        self.x_list.append(x)
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
            p, alpha_init = method(g, x, alpha, hessian, function, gradients) # pass in H or hessian?
            
            # linesearch
            f, g, alpha = linesearch(f, function, g, gradients, x, p, alpha_init, upper_bounds, lower_bounds)
                
            # check if alpha was forced to 0 (due to boundary enforcement)
            if (alpha == 0.0):
                # print('method got stuck on boundary')
                
                # travel along the boundary
                xnew = x+[alpha_init*i for i in p]
                # enforce bounds
                for i in range(len(x)):
                    if (xnew[i] > upper_bounds[i]):
                        xnew[i] = upper_bounds[i]
                    elif (xnew[i] < lower_bounds[i]):
                        xnew[i] = lower_bounds[i]
                        
                if (norm(xnew-x) <  1e-6):
                    print('method got stuck on boundary')
                    break
                
                # update everything
                alpha = np.dot(xnew-x,[alpha_init*i for i in p]) # the projection of the actual step direction in the p direction
                x = xnew
                f = function(x)
                g = gradients(x,function)
            
            else:   
                # update x
                x = x+[alpha*i for i in p]
            
            # store the updated point and associated gradient
            self.x_list.append(x)
            self.f_list.append(f)
            g_list.append(norm(g))
            
            
        self.iterations = method.iters
        self.function_calls = function.counter
        self.solution = x
        self.function_value = f
        self.convergence = g_list