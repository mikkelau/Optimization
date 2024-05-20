# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:51:08 2024

@author: mikke
"""

import optimizer
from MakeContourPlot import MakeContourPlot
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from collections import OrderedDict


class NelderMeadOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.guess = []
        self.x_list = []
        self.f_list = []
        self.current_simplex = []
        self.simplex_list = []
        
    def contour_plot(self):
        if len(self.guess) == 2:
            MakeContourPlot(self.function, self.upper_bounds, self.lower_bounds)
            # plot the best point from each simplex
            plt.plot([i[0] for i in self.x_list],[i[1] for i in self.x_list],c='red',marker='o',markerfacecolor='none')
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n")
            
    def optimize(self, x0, tol=1e-6):
        self.guess = x0
        
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        f_list = []
        
        function.counter = 0
        iters = 0
        
        n = len(x0)
        # create a simplex with edge length l
        simplex = np.empty(shape=(n+1,n),dtype='float64')
        simplex[0] = x0
        # create a dictionary to store vectors and their corresponding function values
        point_to_value = {}
        # define l
        l = 1
        for i in range(1,n+1):
            s = np.empty(2,dtype='float64')
            for j in range(n):
                if j==i:
                    s[j] = (l/n*sqrt(2))*(sqrt(n+1)-1)+l/sqrt(2)
                else:
                    s[j] = (l/n*sqrt(2))*(sqrt(n+1)-1)
            simplex[i] = simplex[0]+s
        # populate the dictionary
        for point in simplex:
            point_to_value[tuple(point)] = function(point)
        simplex_dict = point_to_value
            
        delta_simplex = 0
        for i in range(n):
            delta_simplex += np.linalg.norm(simplex[i]-simplex[n])
            
        while ((delta_simplex>tol) and (iters < max_iters)):
            alpha = 1
            # Order from the lowest (best) to the highest
            simplex_dict = {}
            for point in simplex:
                simplex_dict[tuple(point)] = point_to_value[tuple(point)]
            simplex_dict = OrderedDict(sorted(simplex_dict.items(), key=lambda item: item[1]))
            simplex = np.array([key for key in simplex_dict.keys()])
            
            # save the current best point
            self.x_list.append(simplex[0])

            f_best = list(simplex_dict.values())[0]
            f_worst = list(simplex_dict.values())[-1]
            f_secondworst = list(simplex_dict.values())[-2]
            f_list.append(f_best)
            # print(simplex[0],'\n')
            
            # the centroid excluding the worst point
            summed = 0
            for i in range(n):
                summed += simplex[i]
            x_c = 1/n*summed
            
            # reflection
            x_r = tuple(x_c+alpha*(x_c-simplex[n]))
            if x_r not in point_to_value:
                point_to_value[x_r] = function(x_r)
            f_r = point_to_value[x_r]
            # is reflected point better than the best?
            if f_r < f_best:
                # expand
                alpha *= 2
                x_e = tuple(x_c+alpha*(x_c-simplex[n]))
                if x_e not in point_to_value:
                    point_to_value[x_e] = function(x_e)
                # is expanded point better than the best?
                if point_to_value[x_e] < f_best:
                    # accept expansion and replace worst point
                    simplex[n] = x_e
                else:
                    # accept reflection
                    simplex[n] = x_r
            # is reflected point better than the second worst?
            elif f_r <= f_secondworst:
                # accept reflected point
                simplex[n] = x_r
            else:
                # is reflected point worse that the worst?
                if f_r > f_worst:
                    # inside contraction
                    alpha *= -0.5
                    x_ic = tuple(x_c+alpha*(x_c-simplex[n]))
                    if x_ic not in point_to_value:
                        point_to_value[x_ic] = function(x_ic)
                    # is inside contraction better than the worst?
                    if point_to_value[x_ic] < f_worst:
                        # accept inside contraction
                        simplex[n] = x_ic
                    else:
                        # shrink
                        alpha *= -1
                        for j in range(1,n+1):
                            simplex[j] = simplex[0]+alpha*(simplex[j]-simplex[0])
                else: # reflected point is only better than the worst
                    # outside contraction
                    alpha *= 0.5
                    x_oc = tuple(x_c+alpha*(x_c-simplex[n]))
                    if x_oc not in point_to_value:
                        point_to_value[x_oc] = function(x_oc)
                    # is contraction better than reflection?
                    if point_to_value[x_oc] < f_r:
                        # accept outside contraction
                        simplex[n] = x_oc
                    else:
                        # shrink
                        for j in range(1,n+1):
                            simplex[j] = simplex[0]+alpha*(simplex[j]-simplex[0])
                            
            # increment the iterator
            iters += 1
            
            # add the new point(s) to the master dictionary
            for point in simplex:
                if tuple(point) not in point_to_value:
                    point_to_value[tuple(point)] = function(point)
            
            # re-calculate the convergence criteria
            delta_simplex = 0
            for i in range(n):
                delta_simplex += np.linalg.norm(simplex[i]-simplex[n])
        
        # Order from the lowest (best) to the highest
        simplex_dict = {}
        for point in simplex:
            simplex_dict[tuple(point)] = point_to_value[tuple(point)]
        simplex_dict = OrderedDict(sorted(simplex_dict.items(), key=lambda item: item[1]))
        simplex = np.array([key for key in simplex_dict.keys()])
        
        self.x_list.append(simplex[0])
        self.iterations = iters
        self.function_calls = function.counter
        self.solution = simplex[0]
        self.function_value = function(simplex[0])
        self.convergence = f_list