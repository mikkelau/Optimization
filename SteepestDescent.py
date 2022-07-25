# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:54:31 2022

@author: mikke
"""

from numpy.linalg import norm

def SteepestDescent(g):

    p = [-1*i/norm(g) for i in g]
    
    return p