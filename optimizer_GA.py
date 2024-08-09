# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:43:43 2024

@author: mikke
"""

import optimizer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import random
import time

class GeneticAlgorithmOptimizer(optimizer.Optimizer):
    def __init__(self, function, upper_bounds, lower_bounds, max_iters, plot_generations=False, num_pops=None):
        super().__init__(function, upper_bounds, lower_bounds, max_iters)
        self.x_list = []
        self.f_list = []
        self.plot_generations = plot_generations
        self.num_pops = num_pops
        
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
            
    def optimize(self):
        function = self.function
        max_iters = self.max_iters
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        num_pops = self.num_pops
        
        function.counter = 0
        
        n = len(upper_bounds)
        
        # mutation parameters
        scale = 0.1 #determines the standard deviation of the random numbers generated
        shrink = 0.5 #scalar, [0,1]. As the optimization progress goes forward, decrease the mutation range (for example, shrink?[0.5, 1.0]) is usually used for local search
        mutfrac =2/n # =crossfrac , default in NSGA code
        
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
        if self.plot_generations and n==2:
            fig = self.contour_plot()
            line1, = plt.plot([i[0] for i in points],[i[1] for i in points],c='red',marker='o',markerfacecolor='none',linestyle='none')
            
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(0.3)

            # reset the function counter to 0 so that making the contour plot isn't counted
            function.counter = 0
        
        # determine best fitness and best point
        fitness = np.array([function(point) for point in points])
        idx_best = np.argmin(fitness)
        best = fitness[idx_best]
        best_point = points[idx_best]
        self.f_list.append(best)
        self.x_list.append(points[idx_best])
        
        gen = 1
        while gen <= max_iters:
                        
            # determine the mating pool via tournament selection.
            # each point is randomly paired with another point, and the winner gets added to the mating pool
            pool = []
            pool_fitness = []
            # ensures no mating with oneself (not sure if that's what I want)
            opponents = np.append(random.sample(range(len(points)),len(points)),random.sample(range(len(points)),len(points)))  
            for i in range(0,len(opponents),2):
                if fitness[opponents[i]] < fitness[opponents[i+1]]:
                    pool.append(points[opponents[i]])
                    pool_fitness.append(fitness[opponents[i]])
                else:
                    pool.append(points[opponents[i+1]])
                    pool_fitness.append(fitness[opponents[i+1]])
            
            # crossover
            children = self.create_children(pool,pool_fitness)
                
            # mutation
            # calculate the mutation parameters using scale and shrink
            S = np.array([scale*(1-shrink*gen/max_iters)*(upper_bounds[n]-lower_bounds[n]) for n in range(n)])
            fitness_children = np.empty((num_pops,1))
            mutation_flag = [random.random() < mutfrac for i in range(n)]
            # do the mutation
            for i, child in enumerate(children):
                randNum = np.array([np.random.randn() for j in range(n)])
                child += S*mutation_flag*randNum
                
                # enforce bounds
                child = np.clip(child, lower_bounds, upper_bounds)

                # calculate fitness of the child
                fitness_children[i] = function(child)
                
            # Choose next generation
            parents_with_children = np.vstack((points,children))
            all_fitness = np.append(fitness, fitness_children)
            idx_best = np.argpartition(all_fitness,num_pops)
            fitness = all_fitness[idx_best[:num_pops]]
            points = parents_with_children[idx_best[:num_pops]]
            
            # determine best fitness and best point
            idx_best = np.argmin(fitness)
            best = fitness[idx_best]
            best_point = points[idx_best]
            self.f_list.append(best)
            self.x_list.append(points[idx_best])
            
            # plot current generation
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
        
        self.iterations = gen-1
        self.function_calls = function.counter
        self.solution = best_point
        self.function_value = best
        self.convergence = self.f_list