# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:33:47 2023

@author: mikke
"""
import optimizer
from MakeContourPlot import MakeContourPlot
import matplotlib.pyplot as plt

class GradientBasedOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, x0):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.guess = x0
        print('initial guess:',self.guess)
        self.x_list = None
        self.y_list = None
        
    def contour_plot(self):
        MakeContourPlot(self.function, self.upper_bounds, self.lower_bounds)
        plt.plot(self.x_list,self.y_list,c='red',marker='o',markerfacecolor='none')
            