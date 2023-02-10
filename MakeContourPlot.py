# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 10:58:16 2022

@author: mikke
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def MakeContourPlot(function,upper_lims,lower_lims):
    numpoints = 100
    feature_x = np.linspace(lower_lims[0], upper_lims[0], numpoints)
    feature_y = np.linspace(lower_lims[1], upper_lims[1], numpoints)
    
    # Creating 2-D grid of features
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z = np.empty((numpoints,numpoints))
    for col in range(numpoints):
        for row in range(numpoints):
            Z[row,col] = function((feature_x[col], feature_y[row])) # function inputs defined as a tuple, not sure if that matters
            
    # create filled contour plot
    fig, ax = plt.subplots(1, 1)
    if ((Z.max() > 1000*Z.min()) and (Z.min() > 0)): # check if a logarithmic contour plot is needed
        # maybe if Z.min is less than 0, shift everything up to be able to plot on a logarithmic contour plot?
        ax.contourf(X, Y, Z, 15, locator=ticker.LogLocator())
    else:
        ax.contourf(X, Y, Z, 15)
    
    # plot colorbar
    pcm = ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
    plt.colorbar(pcm,ax=ax)   
    
    # ax.set_title('Filled Contour Plot')
    # ax.set_xlabel('feature_x')
    # ax.set_ylabel('feature_y')
      
