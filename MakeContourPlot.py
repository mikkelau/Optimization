# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 10:58:16 2022

@author: mikke
"""

import matplotlib.pyplot as plt
import numpy as np


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
            
    # plots filled contour plot
    fig, ax = plt.subplots(1, 1)
    ax.contourf(X, Y, Z)
      
    ax.set_title('Filled Contour Plot')
    #ax.set_xlabel('feature_x')
    #ax.set_ylabel('feature_y')
      
