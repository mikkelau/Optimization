# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:33:47 2023

@author: mikke
"""
import optimizer
import matplotlib.pyplot as plt

class GradientBasedOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.guess = []
        self.x_list = []
        self.f_list = []
        
    def contour_plot(self):
        if len(self.guess) == 2:
            self.make_contour_plot(self.function, self.upper_bounds, self.lower_bounds)
            plt.plot([i[0] for i in self.x_list],[i[1] for i in self.x_list],c='red',marker='o',markerfacecolor='none')
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n")

    
            