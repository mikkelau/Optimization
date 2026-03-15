# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:42:49 2026

@author: mikke
"""

import optimizer_GA
import numpy as np
import random
from scipy.stats import qmc

class BinaryEncodedGAOptimizer(optimizer_GA.GeneticAlgorithmOptimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, num_pops=None, plot_generations=False):
        super().__init__(function, upper_bounds, lower_bounds, max_iters, num_pops)
        self.x_list = []
        self.f_list = []
        self.plot_generations = plot_generations
            
    def initialize_pops(self):
        upper_bounds = self.upper_bounds
        num_pops = self.num_pops
        n = len(upper_bounds)
        
        # sample the solution space
        engine = qmc.LatinHypercube(d=n)
        sample = engine.random(n=num_pops)
        
        threshold = 0.5 # 0.5 for equal probability of 0 or 1
        points = (sample >= threshold).astype(int)
    
        return points
    
    def create_children(self,pool,fitness):
        n = len(self.upper_bounds)
        
        children = np.empty((0,n),dtype=int)
        
        # grab parents
        for i in range(0,len(pool),2):
            mom = pool[i]
            dad = pool[i+1]
            
            # determine crossover point
            randIdx = random.randint(1,len(mom)-1)
            
            # single-point crossover
            child1 = np.append(mom[0:randIdx],dad[randIdx:])
            child2 = np.append(dad[0:randIdx],mom[randIdx:])
            
            children = np.vstack((children,np.vstack((child1,child2))))
         
        return children
     
    def mutation(self, children, gen):
        
        # make sure these are all ints
        children = np.array(children,dtype=int)

        n = len(children[0])
        
        # mutation parameters
        mutfrac = 1/n
        
        # determine which bits will flip. Each bit has mutfrac chance of flipping
        mutation_mask = np.random.random(children.shape) < mutfrac # 
        
        # flip the bits
        children[mutation_mask] ^= 1
            
        return children
        