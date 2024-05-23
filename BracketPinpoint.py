# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:42 2022

@author: mikke
"""

from Cubic_Interp import interpolate, plot_linesearch
# from Quadratic_Interp import interpolate, plot_linesearch
#from Bisect import interpolate
import numpy as np
from optimizer_linesearch import LineSearchOptimizer

def linesearch(f_current, function, g, gradients, X, p_dir, alpha, upper_bounds, lower_bounds): 
    # g is the gradient at current point
    # I pass in f_current to avoid another function eval
    
    #print('alpha:', alpha)
        
    mu1 = 1e-4
    mu2 = 0.9
    sigma = 2
    alpha1 = 0.0
    
    alpha2 = alpha
    f1 = f_current
    slope_current = np.dot(g, p_dir) 
    slope1 = slope_current
    first = True
    while True:
        # enforce bounds      
        alpha2, bounds_enforced = LineSearchOptimizer.enforce_bounds(alpha2, X, p_dir, upper_bounds, lower_bounds)
                    
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

def pinpoint(alpha_low, alpha_high, f_current, f_low, f_high, slope_current, slope_low, slope_high, mu1, mu2, function, gradients, X, p_dir):
    k = 0
    while True:
        alpha_p = interpolate(alpha_low, alpha_high, f_low, f_high, slope_low, slope_high)
        Xnew = X+alpha_p*p_dir
        f_p = function(Xnew)
        g_p = gradients(Xnew, function)
        # plot_linesearch(alpha_low, alpha_high, f_low, f_high, slope_low, slope_high, alpha_p, f_p, g_p)
        slope_p = np.dot(g_p, p_dir) # this is the dot product
        if ((f_p > f_current+mu1*alpha_p*slope_current) or (f_p > f_low)): # if new point does not decrease fitness, or if it is not lower than the low step
            alpha_high = alpha_p # make this the "high" point
            # also update f_high, slope_high
            f_high = f_p
            slope_high = slope_p
        else: # fitness was good enough, now check slopes
            if (abs(slope_p) <= -1*mu2*slope_current): # satisfies both strong wolfe criteria
                break # nailed it
            # determine if you need to move alpha_high
            elif (slope_p*(alpha_high-alpha_low) >= 0):
                alpha_high = alpha_low # shrink the bracket
                f_high = f_low
                slope_high = slope_low
            
            # reset alpha_low
            alpha_low = alpha_p 
            f_low = f_p
            slope_low = slope_p
        
        # increment counter
        k += 1 
        if (k >= 3):
            # just take what you have
            break
    
    #print('k:',k)
    return f_p, g_p, alpha_p 

                
        