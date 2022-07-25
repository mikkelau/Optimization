# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:42 2022

@author: mikke
"""

def Backtrack(f_current, function, g, alpha, X, p_dir): 
    # g is the gradient at X
    # I pass in f_current to avoid another function eval
    #print('alpha:', alpha)
    mu = 1e-4
    rho = 0.5
    Xnew = X+[alpha*i for i in p_dir]
    dir_slope = sum([i*j for (i, j) in zip(g, p_dir)]) # this is the dot product
    f_eval = function(Xnew)
    while (f_eval > f_current+mu*alpha*dir_slope):
        alpha = rho*alpha
     #   print('new alpha:',alpha)

        # update Xnew
        Xnew = X+[alpha*i for i in p_dir]
        f_eval = function(Xnew)
    return Xnew, f_eval, alpha