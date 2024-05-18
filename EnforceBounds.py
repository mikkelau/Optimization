# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:30:24 2023

@author: mikke
"""

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