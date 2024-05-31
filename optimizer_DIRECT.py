# -*- coding: utf-8 -*-
"""
Created on Tue May 28 07:40:13 2024

@author: mikke
"""

import optimizer
from MakeContourPlot import MakeContourPlot
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.stats import qmc
from numpy.linalg import norm

class DIRECTOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, tol=1e-6):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        self.tol = tol
        
    def contour_plot(self,points):
        if len(self.upper_bounds) == 2:
            # enable interactive mode
            plt.ion()
            fig = MakeContourPlot(self.function, self.upper_bounds, self.lower_bounds)
            # plot the points that got passed in
            plt.scatter([i[0] for i in points],[i[1] for i in points],edgecolors='r',facecolors='none')
            return fig
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n")  
            
    def find_convex_hull(self, pt_dict):
        # Starting with the point with the lowest distance and best fitness,
        # draw lines to all the minimum points at each distance greater than that of the current point.
        # The other minimum point that results in the lowest slope is a potentially optimal hyperrectangle, and becomes your new starting point.
        # repeat until there are no distances greater than that of your current point.
        # the lowest point at the smallest distance, and the lowest point at the greatest distance will always be potentially optimal hyperrectangles.
        
        S = []

        # sort the dictionary by distance value
        pt_dict2 = {}
        for key in pt_dict:
            pt_dict2[key] = np.array([pt_dict[key][0], norm(pt_dict[key][1])/2])
        pt_dict2 = OrderedDict(sorted(pt_dict2.items(), key=lambda item: item[1][1]))
        
        dist_vals = sorted(set([val[1] for val in pt_dict2.values()]))
        
        dist_dict = {}
        # now, for each unique distance value, find the point with the best fitness
        for val in dist_vals:
            pt_list = [k for k, v in pt_dict2.items() if v[1] == val]
            best_pt = pt_list[0]
            minval = pt_dict2[best_pt][0]
            for pt in pt_list:
                if pt_dict2[pt][0] < minval:
                    best_pt = pt
                    minval = pt_dict2[pt][0]
        
            # this contains the best point at every d value
            dist_dict[val] = np.array([pt_dict2[best_pt][0],best_pt],dtype=object)
            
        dist_dict = OrderedDict(sorted(dist_dict.items()))
        
        while len(dist_vals) > 1:
            d = dist_vals.pop(0)
            # S[dist_dict[d][1]] = pt_dict[dist_dict[d][1]]
            S.append(dist_dict[d][1])
            slope = (dist_dict[dist_vals[0]][0]-dist_dict[d][0])/(dist_vals[0]-d) # rise/run
            keep_dist = dist_vals[0]
            for dist in dist_vals:
                if (dist_dict[dist][0]-dist_dict[d][0])/(dist-d) < slope:
                    slope = (dist_dict[dist][0]-dist_dict[d][0])/(dist-d)
                    keep_dist = dist
            # get rid of all distances between d and keep_dist
            dist_vals = dist_vals[dist_vals.index(keep_dist):]
        
        # grab the last point
        S.append(np.array(dist_dict[dist_vals[-1]][1]))                
        
        return S
            
    def optimize(self):
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        f_list = []
        
        function.counter = 0
        iters = 0
        
        n = len(upper_bounds)
        
        t = np.zeros(n)
        
        # determine the centroid of the search space
        cent = np.array([(lower_bounds[i]+upper_bounds[i])/2 for i in range(len(upper_bounds))])
        
        # make a dictionary where each point is the key, and the value is a vector or list containing fitness and edge lengths (d can be derived from edge lengths)
        pt_dict = {}
        
        f_min = function(cent)
        x_best = cent
        pt_dict[tuple(cent)] = np.array([f_min,np.array([upper-lower for upper,lower in zip(upper_bounds,lower_bounds)])],dtype=object)
        
        min_dist = norm(pt_dict[tuple(cent)][1])
        
        while ((min_dist > self.tol) and (iters < max_iters)):
            self.x_list.append(x_best)
            f_list.append(f_min)
            
            # Find set S of potentially optimal hyperrectangles
            S = self.find_convex_hull(pt_dict)
            
            for pt in S:
                # find the set of dimensions with the maximum side length.
                split_dim = np.where(pt_dict[tuple(pt)][1]==max(pt_dict[tuple(pt)][1]))[0]
                # break ties by selecting the search dimension that has been divided the least over the history of the search.
                if len(split_dim > 1):
                    subset_t = np.array([t[i] for i in split_dim])
                    split_dim_idx = np.where(subset_t==min(subset_t))[0]
                    # If there are still multiple dimensions in the selection, simply select the one with the lowest index
                    split_dim = split_dim[split_dim_idx[0]]
                    
                # Divide the rectangle into thirds along dimension i, creating two new points
                pt_dict[tuple(pt)][1][split_dim] /= 3
                
                pt1 = pt.copy()
                pt1[split_dim] += pt_dict[tuple(pt)][1][split_dim]
                f1 = function(pt1)
                pt_dict[tuple(pt1)] = np.array([f1,pt_dict[tuple(pt)][1]],dtype=object)
                
                pt2 = pt.copy()
                pt2[split_dim] -= pt_dict[tuple(pt)][1][split_dim]
                f2 = function(pt2)
                pt_dict[tuple(pt2)] = np.array([f2,pt_dict[tuple(pt)][1]],dtype=object)
                
                # increment the dimension split counter
                t[split_dim] += 1
                    
                # update f_min, min_dist based on new points   
                if norm(pt_dict[tuple(pt)][1]) < min_dist:
                    min_dist = norm(pt_dict[tuple(pt)][1])
                if f1 < f_min:
                    f_min = f1
                    x_best = pt1
                if f2 < f_min:
                    f_min = f2
                    x_best = pt2

            iters += 1
            
        f_list.append(f_min)
        self.x_list.append(x_best)
        self.iterations = iters
        self.function_calls = function.counter
        self.solution = x_best
        self.function_value = f_min
        self.convergence = f_list