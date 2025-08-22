# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:10:42 2022

@author: mikke
"""
import matplotlib.pyplot as plt 

def interpolate(alpha1, alpha2, f1, f2, slope1, slope2):
           
    denom = 2*(f2-f1+slope1*(alpha1-alpha2))
    
    if denom > 0: # make sure we don't divide by 0
        alpha = (2*alpha1*(f2-f1)+slope1*(alpha1**2-alpha2**2))/denom
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
    
    x = [lower + x*(upper-lower)/(length-1) for x in range(length)]
    
    y = [c0+c1*i+c2*(i**2) for i in x]
    
    plt.plot(x,y)
    plt.plot([lower, alpha, upper], [f_lower, c0+c1*alpha+c2*(alpha**2), f_upper],'o')
    plt.plot(alpha, f_p, 'o', color='red')

    # naming the x axis 
    plt.xlabel('alpha') 
    # naming the y axis 
    plt.ylabel('f(alpha)') 
      
    # function to show the plot 
    plt.show() 