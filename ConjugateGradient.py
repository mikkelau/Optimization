# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:54:31 2022

@author: mikke
"""

from numpy.linalg import norm

def method(g, x, alpha, hessian, function, gradients): # g is a list, not an array
    
    if (abs(sum([i*j for (i, j) in zip(g, method.g_old)])/sum([i*j for (i, j) in zip(g, g)])) >= 0.1): # this will always be true on the first iteration because g and g_old are equal
        method.k = 0 
    if (method.k == 0): # test getting rid of this if statement, just use the one above it. Seems redundant
        p = [-1*i/norm(g) for i in g]
    else:
        print('k =/= 0')
        beta = sum([i*j for (i, j) in zip(g, g)])/sum([i*j for (i, j) in zip(method.g_old, method.g_old)]) 
        p = [j+k for (j, k) in zip([-1*i/norm(g) for i in g], [beta*i for i in method.p_old])]
        
    if method.iters == 0:
        alpha = 1 # this is totally arbitrary, not sure what a good size is
    else:
        alpha = alpha*abs((sum(i*j for i,j in zip(method.g_old, method.p_old))/sum(i*j for i,j in zip(g, p))))

    # update internal quantities
    method.k += 1
    method.p_old = p
    method.g_old = g
    method.iters += 1
    
    return p, alpha