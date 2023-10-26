# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:34:31 2022

@author: mikke
"""
from numpy import sign
from math import sqrt
import matplotlib.pyplot as plt 
from LU_factor import LU_factor

def interpolate(alpha1, alpha2, f1, f2, slope1, slope2):
    
    beta1 = slope1+slope2-3*(f1-f2)/(alpha1-alpha2)
    
    if ((beta1**2) >= slope1*slope2): # make sure we don't take the square root of a negative number
        beta2 = sign(alpha2-alpha1)*sqrt((beta1**2)-slope1*slope2)
        alpha = alpha2-(alpha2-alpha1)*(slope2+beta2-beta1)/(slope2-slope1+2*beta2)
    else:
        alpha = 0.5*(alpha1+alpha2)
        # print('bisected: ill-conditioned')
        
    # make sure interpolated step isn't outside the bracket
    if (((alpha > alpha1) and (alpha > alpha2)) or ((alpha < alpha1) and (alpha < alpha2))):
        # if it is outside, use bisection
        alpha = 0.5*(alpha1+alpha2)
        # print('bisected: outside bracket')
    
    return alpha

def plot_linesearch(alpha1, alpha2, f1, f2, slope1, slope2, alpha, f_p, g_p):

    c2 = slope1/(2*(alpha1-alpha))
    c1 = -2*c2*alpha
    c0 = f1-c1*alpha1-c2*alpha1**2
    
    if alpha1 > alpha2:
        f_upper  = f1
        f_lower = f2
        lower = alpha2
        upper = alpha1
    else:
        f_lower = f1
        f_upper = f2
        lower = alpha1
        upper = alpha2
    length = 100
    
    b = [slope1, slope2, 0]
    A = [[1, 2*alpha1, 3*alpha1**2], [1, 2*alpha2, 3*alpha2**2], [1, 2*alpha, 3*alpha**2]]
    c = LU_factor(b, A)
    c1 = c[0]
    c2 = c[1]
    c3 = c[2]
    c0 = f1-c1*alpha1-c2*(alpha1**2)-c3*(alpha1**3)

    x = [lower + x*(upper-lower)/(length-1) for x in range(length)]
    
    y = [c0+c1*i+c2*(i**2)+c3*(i**3) for i in x]
    
    plt.figure()
    plt.plot(x,y)
    plt.plot([lower, alpha, upper], [f_lower, c0+c1*alpha+c2*(alpha**2)+c3*(alpha**3), f_upper],'o')
    plt.plot(alpha, f_p, 'o', color='red')
    plt.grid()
    
    # naming the x axis 
    plt.xlabel('alpha') 
    # naming the y axis 
    plt.ylabel('f(alpha)') 
      
    # function to show the plot 
    plt.show() 