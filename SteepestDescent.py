# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:54:31 2022

@author: mikke
"""

from numpy.linalg import norm
import numpy as np
import os

def method(g, x, alpha, hessian, function, gradients):
    
    if not hasattr(method, 'name'):
        method.name = os.path.basename(__file__).split('.')[0]

    p = -1*(g/norm(g))

    if (method.iters == 0):
        alpha = 1 # this is totally arbitrary, not sure what a good size is
    else:
        alpha = alpha*abs((np.dot(method.g_old, method.p_old)/np.dot(g, p)))

    # update internal quantities
    method.p_old = p
    method.g_old = g
    method.iters += 1

    return p, alpha