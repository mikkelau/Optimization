# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:18:27 2026

@author: mikke
"""

import optimizer_GA
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import qmc

class RealEncodedGAOptimizer(optimizer_GA.GeneticAlgorithmOptimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, num_pops, plot_generations=False):
        super().__init__(function, upper_bounds, lower_bounds, max_iters, num_pops)
        self.x_list = []
        self.f_list = []
        self.plot_generations = plot_generations
        
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
            
    def initialize_pops(self):
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        num_pops = self.num_pops
        n = len(upper_bounds)
        
        # sample the solution space
        engine = qmc.LatinHypercube(d=n)
        sample = engine.random(n=num_pops)
        
        # if binary == True:
        #     threshold = 0.5 # 0.5 for equal probability of 0 or 1
        #     points = (sample >= threshold).astype(int)
        # else:
        #     points = qmc.scale(sample, lower_bounds, upper_bounds)
            
        points = qmc.scale(sample, lower_bounds, upper_bounds)
        
        return points
    
    # based on NSGA technique
    def create_children(self,pool,fitness):

        n = len(self.upper_bounds)
        ratio = 1.2 # scalar. If it lies in the range [0, 1], the children created are within the two parents. If algorithm is premature, try to set ratio larger than 1.0
        crossfrac = 2/n #default in NSGA code
        
        # generate offspring
        children = np.empty((0,n))
        # create parents
        for i in range(0,len(pool),2):
            mom = pool[i]
            dad = pool[i+1]
            crossover_flag = [random.random() < crossfrac for j in range(len(mom))]
            randNum = np.array([random.random() for j in range(len(mom))])
                        
            # crossover
            child1 = mom+crossover_flag*randNum*ratio*(dad-mom)
            child2 = dad-crossover_flag*randNum*ratio*(dad-mom)
            
            children = np.vstack((children,np.vstack((child1,child2))))
        
        return children
     
     # using the golden ratio
     # def create_children(self,pool,fitness):

     #     phi = (1+5**0.5)/2
         
     #     # generate offspring
     #     children = np.empty((0,2))
     #     # create parents
     #     for i in range(0,len(pool),2):
     #         if fitness[i] < fitness[i+1]:
     #             parent1 = pool[i]
     #             parent2 = pool[i+1]
     #         else:
     #             parent1 = pool[i+1]
     #             parent2 = pool[i]
                         
     #         # crossover
     #         child1 = parent1+(parent2-parent1)/(1+phi)
     #         child2 = parent1-(parent2-parent1)/(1+phi)

     #         children = np.vstack((children,np.vstack((child1,child2))))
         
     #     return children
     
    def mutation(self, children, gen):
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        n = len(upper_bounds)
        
        # mutation parameters
        scale = 0.1 #determines the standard deviation of the random numbers generated
        shrink = 0.5 #scalar, [0,1]. As the optimization progress goes forward, decrease the mutation range (for example, shrink?[0.5, 1.0]) is usually used for local search
        mutfrac = 2/n # =crossfrac , default in NSGA code
                
        # calculate the mutation parameters using scale and shrink
        S = scale*(1-shrink*gen/max_iters)*(np.array(upper_bounds)-np.array(lower_bounds))
        mutation_mask = np.random.random(children.shape) < mutfrac
        
        # do the mutation
        randNum = np.random.randn(*children.shape)
        children += S*randNum*mutation_mask

        # enforce bounds
        children = np.clip(children, lower_bounds, upper_bounds)    
        
        return children