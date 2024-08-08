# -*- coding: utf-8 -*-
"""
Created on Tue May 28 07:40:13 2024

@author: mikke
"""

import optimizer
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import copy

class DIRECTOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, tol=1e-6, eps=1e-4, plot_points=False):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        self.tol = tol
        self.eps = eps
        self.plot_points=plot_points
        
    def contour_plot(self, points=None):
        if len(self.upper_bounds) == 2:
            # enable interactive mode
            plt.ion()
            fig = self.make_contour_plot(self.function, self.upper_bounds, self.lower_bounds)
            # plot the points that got passed in
            if not points:
                points = self.x_list
            plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none')
            return fig
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n")              
    
    def find_convex_hull(self, pt_dict):
        # Starting with the point with the best fitness, draw lines to all the minimum points at each distance greater than that of the current point.
        # The other minimum point that results in the lowest slope is a potentially optimal hyperrectangle, and becomes your new starting point.
        # repeat until there are no distances greater than that of your current point.
        # The lowest point at the greatest distance will always be a potentially optimal hyperrectangle.
        """
        This function finds the lower convex hull of points based on their fitness and distance values.
        Each point is associated with a fitness value and a distance value which is the distance
        from the point to the corner of its hyperrectangle.
        
        Parameters:
        - pt_dict: A dictionary where keys are points and values are a numpy array containing fitness 
                    and another numpy array containing the side lengths of the hyperrectangle.
        
        Returns:
        - S: A list of points forming the lower convex hull.
        """
        
        S = []
    
        # Convert the dictionary to a set of numpy arrays for efficient operations
        points = np.array(list(pt_dict.keys()))
        fitnesses = np.array([pt_dict[tuple(point)][0] for point in points])
        distances = np.array([norm(pt_dict[tuple(point)][1])/2 for point in points],dtype=np.longdouble) # need longdouble to correctly calculate slope
    
        # Sort points by primarily by distance and secondarily by fitness
        sorted_indices = np.lexsort((fitnesses,distances))
        points = points[sorted_indices]
        fitnesses = fitnesses[sorted_indices]
        distances = distances[sorted_indices]
    
        # get rid of any points with distances lower than the distance containing the best fitness
        distance_at_best_pt = np.linalg.norm(pt_dict[tuple(self.x_list[-1])][1]) / 2
        start_index = np.searchsorted(distances, distance_at_best_pt, side='left')
        points = points[start_index:]
        fitnesses = fitnesses[start_index:]
        distances = distances[start_index:]
    
        # Find the best fitness point at each distance
        unique_distances, indices = np.unique(distances, return_index=True)
        best_fitnesses = fitnesses[indices]
        best_points = points[indices]
    
        # convex hull selection
        f_min = self.f_list[-1]
        while len(unique_distances) > 1:
            current_distance = unique_distances[0]
            current_fitness = best_fitnesses[0]
            
            # Calculate the slopes from the current point to the best point at each other distance value
            slopes = (best_fitnesses[1:]-current_fitness)/(unique_distances[1:]-current_distance) # rise/run
            
            # determine the smallest slope
            min_slope_index = np.argmin(slopes)
            slope = slopes[min_slope_index]
            
            # now make sure that the potential optimal point could improve the best fitness by a minimum amount
            if current_fitness-slope*current_distance <= f_min-self.eps*abs(f_min):
                S.append(np.array(best_points[0]))
            
            # get rid of all points between current point and the point resulting in the shallowest slope
            unique_distances = unique_distances[min_slope_index+1:]
            best_fitnesses = best_fitnesses[min_slope_index+1:]
            best_points = best_points[min_slope_index+1:]
    
        # Append the last point
        if len(unique_distances) > 0:
            S.append(np.array(best_points[-1]))
    
        return S
            
    def optimize(self):
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        
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
        pt_dict[tuple(cent)] = np.array([f_min,np.array([upper-lower for upper,lower in zip(upper_bounds,lower_bounds)],dtype=float)],dtype=object)
        # need to define the side lengths as type float, or else problems can happen
        
        min_dist = norm(pt_dict[tuple(cent)][1])/2
        
        while ((min_dist > self.tol) and (iters < max_iters)):
            self.x_list.append(x_best)
            self.f_list.append(f_min)
            
            # Find set S of potentially optimal hyperrectangles
            S = self.find_convex_hull(pt_dict)
            
            for pt in S:
                # find the set of dimensions with the maximum side length.
                split_dim = np.where(pt_dict[tuple(pt)][1]==max(pt_dict[tuple(pt)][1]))[0]
                # break ties by selecting the search dimension that has been divided the least over the history of the search.
                if len(split_dim) > 1:
                    subset_t = np.array([t[i] for i in split_dim])
                    split_dim_idx = np.where(subset_t==min(subset_t))[0]
                    # If there are still multiple dimensions in the selection, simply select the one with the lowest index
                    split_dim = split_dim[split_dim_idx[0]]
                    
                # Divide the rectangle into thirds along dimension i, creating two new points
                pt_dict[tuple(pt)][1][split_dim] /= 3
                
                pt1 = pt.copy() # deep copy
                pt1[split_dim] += pt_dict[tuple(pt)][1][split_dim]
                f1 = function(pt1)
                pt_dict[tuple(pt1)] = np.array([f1,copy.deepcopy(pt_dict[tuple(pt)][1])],dtype=object)
                
                pt2 = pt.copy() # deep copy
                pt2[split_dim] -= pt_dict[tuple(pt)][1][split_dim]
                f2 = function(pt2)
                pt_dict[tuple(pt2)] = np.array([f2,copy.deepcopy(pt_dict[tuple(pt)][1])],dtype=object)
                
                # increment the dimension split counter
                t[split_dim] += 1
                    
                # update f_min, min_dist based on new points   
                if norm(pt_dict[tuple(pt)][1])/2 < min_dist:
                    min_dist = norm(pt_dict[tuple(pt)][1])/2
                if f1 < f_min:
                    f_min = f1
                    x_best = pt1
                if f2 < f_min:
                    f_min = f2
                    x_best = pt2

            iters += 1
            
        # plot the final complex hull
        S = self.find_convex_hull(pt_dict)
        scatter_x = []
        scatter_y = []
        for value in pt_dict.values():
            scatter_x.append(norm(value[1])/2)
            scatter_y.append(value[0])
        x = []
        y = []
        for pt in S:
            x.append(norm(pt_dict[tuple(pt)][1])/2)
            y.append(pt_dict[tuple(pt)][0])
        plt.figure()
        plt.plot(x,y)
        plt.xscale("log") # this may cause the plot of the lower hull to look like it is incorrectly skipping points at some distances
        plt.scatter(scatter_x,scatter_y)
        plt.grid()
        plt.xlabel('d',fontweight='bold')
        plt.ylabel('f',fontweight='bold')
            
        self.x_list.append(x_best) 
        self.f_list.append(f_min)
        self.iterations = iters
        self.function_calls = function.counter
        self.solution = x_best
        self.function_value = f_min
        self.convergence = self.f_list
        
        # plot all the points tested
        if self.plot_points and n==2:
            # enable interactive mode
            # plt.ion()
            MakeContourPlot(self.function, self.upper_bounds, self.lower_bounds)
            points = np.array(list(pt_dict.keys()))
            plt.scatter([i[0] for i in points],[i[1] for i in points], edgecolors='r',facecolors='none')