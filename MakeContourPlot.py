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
        cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator())
        # plot colorbar
        cbar = fig.colorbar(cs, format='%.0e')
    else:
        cs = ax.contourf(X, Y, Z, 16)
        # plot colorbar
        cbar = fig.colorbar(cs)
        
    # set the axes to be on the same scale
    ax.set_aspect('equal', adjustable='box')  
    
    cfm = plt.get_current_fig_manager()
    cfm.window.activateWindow()
    cfm.window.raise_()
        
    # ax.set_title('Filled Contour Plot')
    # ax.set_xlabel('feature_x')
    # ax.set_ylabel('feature_y')
    
    return fig