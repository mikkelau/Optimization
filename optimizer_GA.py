# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:43:43 2024

@author: mikke
"""

import optimizer
from MakeContourPlot import MakeContourPlot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import random
import time

class GeneticAlgorithmOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, plot_generations=False):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        self.plot_generations = plot_generations
        
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
        
        function.counter = 0
        
        n = len(upper_bounds)
        
        ratio = 1.2 # scalar. If it lies in the range [0, 1], the children created are within the two parents. If algorithm is premature, try to set ratio larger than 1.0
        crossfrac = 2/n #default in NSGA code
        
        # mutation
        scale = 0.1 #determines the standard deviation of the random numbers generated
        shrink = 0.5 #scalar, [0,1]. As the optimization progress goes forward, decrease the mutation range (for example, shrink?[0.5, 1.0]) is usually used for local search
        mutfrac = crossfrac # =2/n, default in NSGA code
        
        #create first generation
        pop = 15*n # should be 15-20 x number of design variables
        #needs to be even
        if (pop%2 == 1):
            pop+=1
        # pop/2 should also be even
        if ((pop/2)%2 == 1):
            pop+=2
            
        # sample the solution space
        engine = qmc.LatinHypercube(d=n)
        sample = engine.random(n=pop)
        points = qmc.scale(sample, lower_bounds, upper_bounds)
        fitness = np.array([function(point) for point in points])
        
        # plot initial population
        if self.plot_generations and n==2:
            fig = self.contour_plot()
            line1, = plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none',linestyle='none')
            
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.3)

            # reset the function counter to 0 so that making the contour plot isn't counted
            function.counter = 0
        
        gen = 1
        while gen <= max_iters:
            
            # determine best fitness and best point
            idx_best = np.argmin(fitness)
            best = fitness[idx_best]
            best_point = points[idx_best]
            self.f_list.append(best)
            self.x_list.append(points[idx_best])
            
            # determine the mating pool via tournament selection.
            # each point is randomly paired with another point, and the winner gets added to the mating pool
            pool = []
            opponents = np.append(random.sample(range(len(points)),len(points)),random.sample(range(len(points)),len(points)))   
            for i in range(0,len(opponents),2):
                if fitness[opponents[i]] < fitness[opponents[i+1]]:
                    pool.append(points[opponents[i]])
                else:
                    pool.append(points[opponents[i+1]])
                    
            # generate offspring
            children = np.empty((0,2))
            # crossover
            # create parents
            for i in range(0,len(pool),2):
                mom = pool[i]
                dad = pool[i+1]
                crossover_flag = [random.random() < crossfrac for n in range(n)]
                randNum = [random.random() for n in range(n)]
                #child1 = np.empty((1,2))
                #child2 = np.empty((1,2))
                child1 = []
                child2 = []
                
                for j in range(n):
                    # child 1
                    child1.append(mom[j]+crossover_flag[j]*randNum[j]*ratio*(dad[j]-mom[j]))
                    # enforce bounds
                    if child1[j] < lower_bounds[j]:
                        child1[j] = lower_bounds[j]
                    elif child1[j] > upper_bounds[j]:
                        child1[j] = upper_bounds[j]
                    # child 2
                    #child2[j] = dad[j]-crossover_flag[j]*randNum[j]*ratio*(dad[j]-mom[j])
                    child2.append(dad[j]-crossover_flag[j]*randNum[j]*ratio*(dad[j]-mom[j]))
                    # enforce bounds
                    if child2[j] < lower_bounds[j]:
                        child2[j] = lower_bounds[j]
                    elif child2[j] > upper_bounds[j]:
                        child2[j] = upper_bounds[j]
                #children.append(child1)
                #children.append(child2)
                children = np.vstack((children,child1))
                children = np.vstack((children,child2))
                
            # Mutate
            # calculate the mutation parameters using scale and shrink
            S = [scale*(1-shrink*gen/max_iters)*(upper_bounds[n]-lower_bounds[n]) for n in range(n)]
            fitness_children = np.empty((pop,1))
            # do the mutation
            for i, child in enumerate(children):
                for j in range(n):
                    if random.random() < mutfrac:
                        child[j] = child[j]+S[j]*np.random.randn()
                        # enforce bounds
                        if child[j] < lower_bounds[j]:
                            child[j] = lower_bounds[j]
                        elif child[j] > upper_bounds[j]:
                            child[j] = upper_bounds[j]
                # determine fitness of each child
                fitness_children[i] = function(child)
                
            # Choose next generation
            parents_with_children = np.vstack((points,children))
            all_fitness = np.append(fitness, fitness_children)
            idx_best = np.argpartition(all_fitness,pop)
            fitness = all_fitness[idx_best[:pop]]
            points = parents_with_children[idx_best[:pop]]
            
            # plot current simplex
            if self.plot_generations and n==2:
                # updating the values of the simplex
                line1.set_xdata([i[0] for i in np.vstack(points)])
                line1.set_ydata([i[1] for i in np.vstack(points)])
                # re-drawing the figure
                fig.canvas.draw()
                # to flush the GUI events
                fig.canvas.flush_events()
                time.sleep(0.1)
                
            # increment generation count
            gen+=1
        
        idx_best = np.argmin(fitness)
        best = fitness[idx_best]
        best_point = points[idx_best]
        self.f_list.append(best)
        self.x_list.append(points[idx_best])
        self.iterations = gen
        self.function_calls = function.counter
        self.solution = best_point
        self.function_value = best
        self.convergence = self.f_list