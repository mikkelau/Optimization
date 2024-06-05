# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:51:08 2024

@author: mikke
"""

import optimizer
from MakeContourPlot import MakeContourPlot
import numpy as np
from math import sqrt
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm

class NelderMeadOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, tol=1e-6, plot_simplex=False):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.guess = []
        self.x_list = []
        self.f_list = []
        self.current_simplex = []
        self.tol = tol
        self.plot_simplex = plot_simplex
        
    def contour_plot(self,points):
        if len(self.guess) == 2:
            # enable interactive mode
            plt.ion()
            fig = MakeContourPlot(self.function, self.upper_bounds, self.lower_bounds)
            # plot the points that got passed in
            line1, = plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none')
            return fig,line1
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n")  
            
    def enforce_bounds(self, X, x_c, alpha):
        bounds_enforced = False
        p_dir = (x_c-X)
        for i in range(len(X)):
            if (x_c[i]+alpha*p_dir[i] > self.upper_bounds[i]):
                # solve for the alpha that would land on the boundary
                alpha_new = (self.upper_bounds[i]-x_c[i])/p_dir[i]
                if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                    alpha = alpha_new
                    bounds_enforced = True
            elif (x_c[i]+alpha*p_dir[i] < self.lower_bounds[i]):
                # solve for the alpha that would land on the boundary
                alpha_new = (self.lower_bounds[i]-x_c[i])/p_dir[i]
                if (alpha_new < alpha): # this check is needed to make sure we aren't overwriting an alpha that was already solved for when checking a different bound
                    alpha = alpha_new
                    bounds_enforced = True
                    
        Xnew = tuple(x_c+alpha*(x_c-X))
        
        return Xnew, alpha, bounds_enforced
            
    def optimize(self, x0):
        self.guess = x0
        
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        tol = self.tol
        f_list = []
        
        function.counter = 0
        iters = 0
        
        # enforce bounds in initial guess
        guess_enforced = False
        for i in range(len(x0)):
            if x0[i] > upper_bounds[i]:
                x0[i] = upper_bounds[i]
                guess_enforced = True
            elif x0[i] < lower_bounds[i]:
                x0[i] = lower_bounds[i]
                guess_enforced = True
        if guess_enforced == True:
            print('Bounds enforced for initial guess')
        
        n = len(x0)
        # create a simplex with edge length l
        simplex = np.empty(shape=(n+1,n),dtype='float64')
        simplex[0] = x0
        # define l
        ranges = np.array([i-j for i,j in zip(upper_bounds,lower_bounds)])
        l = 0.25*min(ranges) # this is arbitrary, it might be good to determine this using some other criteria
        if n==2:
            # determine the centroid of the search space
            cent = [(lower_bounds[i]+upper_bounds[i])/2 for i in range(len(upper_bounds))]
            # find the direction of the centroid from x0
            p = np.array([j-i for i,j in zip(x0,cent)])
            # determine the location of the centroid of the remaining two points,
            # constraining it to be in the direction of the centroid of the search space
            c_pts = x0+np.array([i/norm(p) for i in p])*l*sqrt(n*(n+1)/2)/n
            simplex[1] = [c_pts[0]+p[1]/norm(p)*l/2,c_pts[1]-p[0]/norm(p)*l/2]
            simplex[2] = [c_pts[0]-p[1]/norm(p)*l/2,c_pts[1]+p[0]/norm(p)*l/2]
            # is there a way to generalize this for a tetrehedron or hypertetrahedron?
        else:
            for i in range(1,n+1):
                s = np.empty(n,dtype='float64')
                j = 0
                while j < n:
                    if j==i:
                        s[j] = (l/n*sqrt(2))*(sqrt(n+1)-1)+l/sqrt(2)
                    else:
                        s[j] = (l/n*sqrt(2))*(sqrt(n+1)-1)
                    # ensure no starting points are outside of bounds
                    if (simplex[0,j]+s[j] > upper_bounds[j]) or (simplex[0,j]+s[j] < lower_bounds[j]):
                        l *= 0.9
                        j = 0
                    else:
                        j += 1
                simplex[i] = simplex[0]+s
                
        # plot current simplex
        if self.plot_simplex and n==2:
            fig,line1 = self.contour_plot(np.vstack([simplex, simplex[0]]))
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.3)

            # reset the function counter to 0 so that making the contour plot isn't counted
            function.counter = 0
            
        # create a dictionary to store points and their corresponding function values
        point_to_value = {}
        
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
            x_r, alpha, bounds_enforced = self.enforce_bounds(simplex[n], x_c, alpha)
            
            if x_r not in point_to_value:
                point_to_value[x_r] = function(x_r)
            f_r = point_to_value[x_r]
            # is reflected point better than the best?
            if f_r < f_best:
                if bounds_enforced:
                    # accept reflection
                    simplex[n] = x_r
                else:
                    # expand
                    alpha *= 2
                    x_e, alpha, bounds_enforced = self.enforce_bounds(simplex[n], x_c, alpha)
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
                    alpha = -0.5
                    x_ic = tuple(x_c+alpha*(x_c-simplex[n]))
                    if x_ic not in point_to_value:
                        point_to_value[x_ic] = function(x_ic)
                    # is inside contraction better than the worst?
                    if point_to_value[x_ic] < f_worst:
                        # accept inside contraction
                        simplex[n] = x_ic
                    else:
                        # shrink
                        for j in range(1,n+1):
                            simplex[j] = simplex[0]+0.5*(simplex[j]-simplex[0])
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
                            simplex[j] = simplex[0]+0.5*(simplex[j]-simplex[0])
                            
            # increment the iterator
            iters += 1
            
            # add the new point(s) to the master dictionary
            for point in simplex:
                if tuple(point) not in point_to_value:
                    point_to_value[tuple(point)] = function(point)
                    
            # plot current simplex
            if self.plot_simplex and n==2:
                # updating the values of the simplex
                line1.set_xdata([i[0] for i in np.vstack([simplex, simplex[0]])])
                line1.set_ydata([i[1] for i in np.vstack([simplex, simplex[0]])])
                # re-drawing the figure
                fig.canvas.draw()
                # to flush the GUI events
                fig.canvas.flush_events()
                time.sleep(0.1)
            
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
        f_list.append(list(simplex_dict.values())[0])
        
        self.x_list.append(simplex[0])
        self.iterations = iters
        self.function_calls = function.counter
        self.solution = simplex[0]
        self.function_value = function(simplex[0])
        self.convergence = f_list