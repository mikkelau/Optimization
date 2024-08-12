# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:53:11 2024

@author: mikke
"""

import optimizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from random import random, seed
import time
import copy
from numpy.linalg import norm

class ParticleSwarmOptimizer(optimizer.Optimizer):
    def __init__(self, function,upper_bounds,lower_bounds,max_iters,num_pops=None,alpha=1.0,max_beta=1.5,max_gamma=1.5,max_delta=None,plot_swarm=False,seed_num=[]):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        self.num_pops = num_pops
        self.alpha = alpha
        self.max_beta = max_beta
        self.max_gamma = max_gamma
        self.max_delta = max_delta
        self.plot_swarm = plot_swarm
        self.seed_num = seed_num
        
    def contour_plot(self,points=None):
        if len(self.upper_bounds) == 2:
            if points is None:
                points = self.x_list
            # enable interactive mode
            plt.ion()
            fig = self.make_contour_plot(self.function, self.upper_bounds, self.lower_bounds)
            # plot the points that got passed in
            plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none')
            return fig
        else:
            print("Cannot create contour plot. Number of independent variables needs to be two.\n") 
            
    def optimize(self):
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        num_pops = self.num_pops
        alpha = self.alpha
        max_beta = self.max_beta
        max_gamma = self.max_gamma
        max_delta = self.max_delta
        
        function.counter = 0
        
        n = len(upper_bounds)
        
        # determine max_delta
        if not max_delta:
            max_delta = norm(0.02*(np.array(upper_bounds)-np.array(lower_bounds))) # this is arbitrary, maybe spend some time tuning
        
        #create first generation
        if num_pops:
            #needs to be even
            if (num_pops%2 == 1):
                num_pops+=1
        else:
            num_pops = 16*n # should be 15-20 x number of design variables
        
        # sample the solution space
        if self.seed_num:
            seed(self.seed_num)
            engine = qmc.LatinHypercube(d=n,seed=self.seed_num)
        else:
            engine = qmc.LatinHypercube(d=n)
        sample = engine.random(n=num_pops)
        points = qmc.scale(sample, lower_bounds, upper_bounds)
        
        # determine best fitness and best point
        fitness = np.array([function(point) for point in points])
        idx_best = np.argmin(fitness) # what if there are multiple points with the best fitness?
        f_best = min(fitness)
        x_best = copy.deepcopy(points[idx_best])
        self.f_list.append(f_best)
        self.x_list.append(x_best)
        
        # initialize velocity, historical best point, and historical best fitness arrays 
        velocities = np.zeros((len(points),n))
        particle_best_point = qmc.scale(sample, lower_bounds, upper_bounds)
        particle_best_fitness = copy.deepcopy(fitness)
        
        # plot initial population
        if self.plot_swarm and n==2:
            # enable interactive mode
            plt.ion()
            fig = self.make_contour_plot(self.function, self.upper_bounds, self.lower_bounds)
            line1, = plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none',linestyle='none')
            # plot best point
            line2, = plt.plot([x_best[0]],[x_best[1]],c='green',marker='o',markerfacecolor='none',linestyle='none')
            
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.3)

            # reset the function counter to 0 so that making the contour plot isn't counted
            function.counter = 0
        
        iters = 0
        while iters < max_iters:
            
            for i in range(len(points)):
                
                # determine weights
                beta = random()*max_beta
                gamma = random()*max_gamma
                
                # calulate velocity
                velocities[i] = alpha*velocities[i]+beta*(particle_best_point[i]-points[i])+gamma*(x_best-points[i])
                # limit velocity
                if norm(velocities[i]) > max_delta:
                    velocities[i] = max_delta*velocities[i]/norm(velocities[i])
                # update particle position
                points[i] += velocities[i]
                # enforce bounds
                points[i] = np.clip(points[i], lower_bounds, upper_bounds)
                fitness[i] = function(points[i])
                
                # see if particle best needs to be updated
                if fitness[i] < particle_best_fitness[i]:
                    particle_best_point[i] = points[i]
                    particle_best_fitness[i] = fitness[i]
                # see if swarm best needs to be updated
                if fitness[i] < f_best:
                   x_best = copy.deepcopy(points[i])
                   f_best = fitness[i]
                   
            # determine best fitness and best point
            self.f_list.append(f_best)
            self.x_list.append(x_best)

            # plot current locations
            if self.plot_swarm and n==2:
                # updating the swarm locations
                line1.set_xdata([i[0] for i in np.vstack(points)])
                line1.set_ydata([i[1] for i in np.vstack(points)])
                line2.set_xdata(x_best[0])
                line2.set_ydata(x_best[1])
                # re-drawing the figure
                fig.canvas.draw()
                # to flush the GUI events
                fig.canvas.flush_events()
                time.sleep(0.1)
                    
            # increment the iteration count        
            iters += 1
   
        self.iterations = iters
        self.function_calls = function.counter
        self.solution = x_best
        self.function_value = f_best
        self.convergence = self.f_list