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
        alpha_new = alpha
        for i in range(len(X)):
            if (X[i]+alpha*p_dir[i] > upper_bounds[i]):
                # solve for the alpha that would land on the boundary
                alpha_boundary = (upper_bounds[i]-X[i])/p_dir[i]
                alpha_new = min(alpha_new,alpha_boundary) # this makes sure we aren't overwriting an alpha that was already solved for when checking a different bound
            elif (X[i]+alpha*p_dir[i] < lower_bounds[i]):
                # solve for the alpha that would land on the boundary
                alpha_boundary = (lower_bounds[i]-X[i])/p_dir[i]
                alpha_new = min(alpha_new,alpha_boundary) # this makes sure we aren't overwriting an alpha that was already solved for when checking a different bound
        if alpha_new < alpha:
            bounds_enforced = True
            alpha = alpha_new
        
        return alpha, bounds_enforced
    
    def optimize(self, method, linesearch, x0, gradients=FiniteDifference.gradients, hessian=FiniteDifference.hessian):
        self.guess = x0
        
        g_list = []
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        
        function.counter = 0
        method.iters = 0
        
        # enforce bounds in initial guess
        guess_enforced = False
        for i in range(len(x0)):
            if x0[i] > upper_bounds[i]:
                x0[i] = upper_bounds[i]
                guess_enforced = True
            elif x0[i] < lower_bounds[i]:
                x0[i] = lower_bounds[i]
                guess_enforced = True
        if guess_enforced == True:
            print('Bounds enforced for initial guess')
        
        self.x_list.append(x0)
        
        f = function(x0)
        g = gradients(x0,function)
        g_list.append(norm(g))
        alpha = 1
        x = x0

        while ((norm(g) > 1e-6) and (method.iters < max_iters)):
            
            # choose a search direction. should pass out a search direction and initial guess for alpha
            p, alpha_init = method(g, x, alpha, hessian, function, gradients) # pass in H or hessian?
            
            # linesearch
            f, g, alpha = linesearch(f, function, g, gradients, x, p, alpha_init, upper_bounds, lower_bounds)
                
            # check if alpha was forced to 0 (due to boundary enforcement)
            if (alpha == 0.0):
                
                # travel along the boundary by enforcing bounds
                xnew = x+[alpha_init*i for i in p]
                # enforce bounds
                xnew = np.clip(xnew, lower_bounds, upper_bounds)
                
                # recalculate step direction along boundary
                p = xnew-x
                
                # make sure step is greater than 0
                if np.array_equal(x, xnew):
                    print('method got stuck on boundary')
                    break
                
                # make sure the slope in the new p direction is negative
                if np.dot(g, p) > 0:
                    print('method got stuck on boundary')
                    break

                # redo the line search
                f, g, alpha = linesearch(f, function, g, gradients, x, p, alpha_init, upper_bounds, lower_bounds)
                
                # if you're taking really small steps, just quit
                if alpha <= (np.finfo(np.float32).eps)**(1/3):
                    print('method got stuck on boundary')
                    break               
            
            # update x
            x = x+[alpha*i for i in p]
            
            # store the updated point and associated gradient
            self.x_list.append(x)
            self.f_list.append(f)
            g_list.append(norm(g))
            
        self.iterations = method.iters
        self.function_calls = function.counter
        self.solution = x
        self.function_value = self.f_list[-1]
        self.convergence = g_list