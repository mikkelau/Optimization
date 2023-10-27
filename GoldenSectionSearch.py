# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 08:57:52 2023

@author: mikke
"""

import numpy as np
from EnforceBounds import enforce_bounds

def linesearch(f_current, function, g, gradients, X, p_dir, alpha, upper_bounds, lower_bounds): 
    # g is the gradient at current point
    # I pass in f_current to avoid another function eval
    
    #print('alpha:', alpha)
        
    mu1 = 1e-4
    mu2 = 0.9
    phi = (1+5**0.5)/2
    sigma = 1+phi
    alpha1 = 0.0
    
    alpha2 = alpha
    f1 = f_current
    slope_current = np.dot(g, p_dir) 
    slope1 = slope_current
    first = True
    
    # bracket
    while True:
        # enforce bounds      
        alpha2, bounds_enforced = enforce_bounds(alpha2, X, p_dir, upper_bounds, lower_bounds)
                    
        Xnew = X+alpha2*p_dir
        f2 = function(Xnew)
        f_eval = f2
        g2 = gradients(Xnew, function)
        g_eval = g2
        slope2 = np.dot(g2, p_dir) # this is the dot product
        if (f2 > f_current+mu1*alpha2*slope_current) or (not first and (f2 > f1)): # new point is worse than current point
            # There is a point that satisfies strong wolfe conditions between current and guess, so find it
            f_eval, g_eval, alpha = pinpoint(alpha1, alpha2, f_current, f1, f2, slope_current, slope1, slope2, mu1, mu2, function, gradients, X, p_dir)            
            break
        # otherwise, your guess point satisfies the first strong wolfe condition, so check if it satisfies the second
        if (abs(slope2) <= -1*mu2*slope_current): # if the slope at the guess location satisfies strong wolfe condition
            alpha = alpha2 
            break
        # otherwise, check which: you have overshot or undershot the point that you want
        elif (slope2 >= 0): # you have overshot the good point, so you know the good point exists between current point and guess
            f_eval, g_eval, alpha = pinpoint(alpha2, alpha1, f_current, f2, f1, slope_current, slope2, slope1, mu1, mu2, function, gradients, X, p_dir)
            break
        # how can it get stuck here? If alpha passed in is 0!
        else: # you are still moving downward at a high slope, extend your guess
            # if the bounds were enforced, alpha2 is as big as it can get
            if (bounds_enforced == True): # take the alpha on the bounds
                alpha = alpha2
                break
            else:
                alpha1 = alpha2
                alpha2 = sigma*alpha2
        first = False
    
    return f_eval, g_eval, alpha

def pinpoint(alpha1, alpha4, f_current, f1, f4, slope_current, slope_low, slope_high, mu1, mu2, function, gradients, X, p_dir):
    phi = (1+5**0.5)/2
    while True:
        # add a check to see if the interval is already too small?
        
        # calulate interior point
        alpha2 = (alpha4-alpha1)/(1+phi)
        X2 = X+alpha2*p_dir
        f2 = function(X2)
        # check if it satisfies the first wolfe condition
        if (f2 < f_current+mu1*alpha2*slope_current):
            g2 = gradients(X2, function)
            slope2 = np.dot(g2, p_dir)
            # check if it satisfies the second wolfe condition
            if (abs(slope2) <= -1*mu2*slope_current): # satisfies both strong wolfe criteria
                break # nailed it
    