# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 08:57:52 2023

@author: mikke
"""

import numpy as np
from optimizer_linesearch import LineSearchOptimizer
from numpy.linalg import norm

def linesearch(f_current, function, g, gradients, X, p_dir, alpha, upper_bounds, lower_bounds, min_step, method_name): 
    # g is the gradient at current point
    # I pass in f_current to avoid another function eval
    
    #print('alpha:', alpha)
        
    mu1 = 1e-4
    mu2 = 0.7
    alpha1 = 0.0
    
    alpha4 = alpha
    f1 = f_current
    slope_current = np.dot(g, p_dir) 
    first = True
    
    # bracket
    while True:
        # enforce bounds      
        alpha4, Xnew, bounds_enforced = LineSearchOptimizer.enforce_bounds(alpha4, X, p_dir, upper_bounds, lower_bounds)
        
        f4 = function(Xnew)
        f_eval = f4
        g4 = gradients(Xnew, function)
        g_eval = g4
        slope4 = np.dot(g4, p_dir) # this is the dot product
        
        # see if bound enforcement caused step size to go below min_step
        if norm(alpha4*p_dir) <= min_step and bounds_enforced: 
            alpha = alpha4
            break
        if (f4 > f_current+mu1*alpha4*slope_current) or (not first and (f4 > f1)): # new point is worse than current point
            # There is a point that satisfies strong wolfe conditions between current and guess, so find it
            f_eval, g_eval, alpha, Xnew = pinpoint(alpha1, alpha4, f_current, slope_current, mu1, mu2, function, gradients, X, p_dir, min_step)            
            break
        # otherwise, your guess point satisfies the first strong wolfe condition, so check if it satisfies the second
        if (abs(slope4) <= -1*mu2*slope_current): # if the slope at the guess location satisfies strong wolfe condition
            alpha = alpha4 
            break
        # otherwise, check which: you have overshot or undershot the point that you want
        elif (slope4 >= 0): # you have overshot the good point, so you know the good point exists between current point and guess
            f_eval, g_eval, alpha, Xnew = pinpoint(alpha1, alpha4, f_current, slope_current, mu1, mu2, function, gradients, X, p_dir, min_step)
            break
        # how can it get stuck here? If alpha passed in is 0!
        else: # you are still moving downward at a high slope, extend your guess
            # if the bounds were enforced, alpha4 is as big as it can get
            if (bounds_enforced == True): # take the alpha on the bounds
                alpha = alpha4
                break
            else:
                alpha1 = alpha4
                f1 = f4
                alpha4 = 2*alpha4 # could extend by using phi, but testing showed that just doubling the step usually worked best
        first = False
    
    return f_eval, g_eval, alpha, Xnew

def pinpoint(alpha1, alpha4, f_current, slope_current, mu1, mu2, function, gradients, X, p_dir, min_step):
    phi = (1+5**0.5)/2
    k = 0
    alpha2 = None
    alpha3 = None
    while True:
        # enforce minimum step on upper alpha. Sometimes the step can get too small, especially when using estimations of the gradient
        if norm(alpha4*p_dir) < min_step:
            # print("minimum step enforced")
            alpha_p = min_step/norm(p_dir)
            Xnew = X+alpha_p*p_dir
            f_p = function(Xnew)
            g_p = gradients(Xnew, function)
            break
        
        if alpha2==None:
            # calulate first interior point
            alpha2 = alpha1+(alpha4-alpha1)/(1+phi)
            X2 = X+alpha2*p_dir
            f2 = function(X2)
            # check if it satisfies the first wolfe condition
            if (f2 <= f_current+mu1*alpha2*slope_current):
                g2 = gradients(X2, function)
                slope2 = np.dot(g2, p_dir)
                # check if it satisfies the second wolfe condition
                if (abs(slope2) <= -1*mu2*slope_current): # satisfies both strong wolfe criteria
                    f_p = f2
                    g_p = g2
                    alpha_p = alpha2
                    Xnew = X2
                    break # nailed it
        if alpha3==None: # if not
            # calculate second interior point
            alpha3 = alpha1+(alpha2-alpha1)*(1+1/phi)
            X3 = X+alpha3*p_dir
            f3 = function(X3)
            # check if it satisfies the first wolfe condition
            if (f3 <= f_current+mu1*alpha3*slope_current):
                g3 = gradients(X3, function)
                slope3 = np.dot(g3, p_dir)
                # check if it satisfies the second wolfe condition
                if (abs(slope3) <= -1*mu2*slope_current): # satisfies both strong wolfe criteria
                    f_p = f3
                    g_p = g3
                    alpha_p = alpha3
                    Xnew = X3
                    break # nailed it
        
        # if too many tries in the loop, just take the best of what you have.
        if k > 6:
            if (f2>f3):
                Xnew = X+alpha3*p_dir
                f_p = f3
                g_p = gradients(Xnew, function)
                alpha_p = alpha3
                break
            else:
                Xnew = X+alpha2*p_dir
                f_p = f2
                g_p = gradients(Xnew, function)
                alpha_p = alpha2
                break
        
        # shrink the window
        if (f2>f3):
            alpha1 = alpha2
            alpha2 = alpha3
            f2 = f3
            alpha3 = None
            f3 = None
        else:
            alpha4 = alpha3
            alpha3 = alpha2
            f3 = f2
            alpha2 = None
            f2 = None
            
        # increment counter
        k +=1
    
    # enforce minimum step on alpha_p
    if norm(alpha_p*p_dir) < min_step:
        # print("minimum step enforced")
        alpha_p = min_step/norm(p_dir)
        Xnew = X+alpha_p*p_dir
        f_p = function(Xnew)
        g_p = gradients(Xnew, function)
        
    return f_p, g_p, alpha_p, Xnew 