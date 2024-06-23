# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:53:11 2024

@author: mikke
"""

import optimizer
from MakeContourPlot import MakeContourPlot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import random
import time
import copy

class ParticleSwarmOptimizer(optimizer.Optimizer):
    def __init__(self, function,upper_bounds,lower_bounds,max_iters,num_pops=None,alpha=1.0,max_beta=1.5,max_gamma=1.5,max_delta=None,plot_particles=False):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        self.num_pops = num_pops
        self.alpha = alpha
        self.max_beta = max_beta
        self.max_gamma = max_gamma
        self.max_delta = max_delta
        self.plot_particles = plot_particles
        
    def contour_plot(self,points=[]):
        if len(self.upper_bounds) == 2:
            # enable interactive mode
            plt.ion()
            fig = MakeContourPlot(self.function, self.upper_bounds, self.lower_bounds)
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
            max_delta = 0.02*(np.array(upper_bounds)-np.array(lower_bounds)) # this is arbitrary, maybe spend some time tuning
        
        #create first generation
        if num_pops:
            #needs to be even
            if (num_pops%2 == 1):
                num_pops+=1
        else:
            num_pops = 16*n # should be 15-20 x number of design variables
        
        # sample the solution space
        engine = qmc.LatinHypercube(d=n)
        sample = engine.random(n=num_pops)
        points = qmc.scale(sample, lower_bounds, upper_bounds)
        
        # plot initial population
        if self.plot_particles and n==2:
            fig = self.contour_plot()
            line1, = plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none',linestyle='none')
            
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.3)

            # reset the function counter to 0 so that making the contour plot isn't counted
            function.counter = 0
            
        # determine best fitness and best point
        fitness = np.array([function(point) for point in points])
        idx_best = np.argmin(fitness) # what if there are multiple points with the best fitness?
        f_best = fitness[idx_best]
        x_best = points[idx_best]
        self.f_list.append(f_best)
        self.x_list.append(points[idx_best])
        
        # initialize velocity, historical best point, and historical best fitness arrays 
        velocities = np.zeros((len(points),n))
        particle_best_point = qmc.scale(sample, lower_bounds, upper_bounds)
        particle_best_fitness = copy.copy(fitness) # not sure if I need to call deep copy here
        
        iters = 0
        while iters < max_iters:
            
            for i in range(len(points)):
                # see if particle best needs to be updated
                if fitness[i] < particle_best_fitness[i]:
                    particle_best_point[i] = points[i]
                    particle_best_fitness[i] = fitness[i]
                # see if swarm best needs to be updated
                if fitness[i] < f_best:
                   x_best = points[i]
                   f_best = fitness[i]
                   
                # determine weights
                beta = random.random()*max_beta
                gamma = random.random()*max_gamma
                
                # calulate velocity
                velocities[i] = alpha*velocities[i]+beta*(particle_best_point[i]-points[i])+gamma*(x_best-points[i])
                # limit velocity
                velocities[i] = np.clip(velocities[i],-max_delta,max_delta)
                # update particle position
                points[i] += velocities[i]
                # enforce bounds
                points[i] = np.clip(points[i], lower_bounds, upper_bounds)
                fitness[i] = function(points[i])
                
            # determine best fitness and best point
            self.f_list.append(f_best)
            self.x_list.append(x_best)
                
            # plot current locations
            if self.plot_particles and n==2:
                # updating the values of the simplex
                line1.set_xdata([i[0] for i in np.vstack(points)])
                line1.set_ydata([i[1] for i in np.vstack(points)])
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