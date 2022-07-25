# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 11:31:55 2022

@author: mikke
"""

from BeanFunction import BeanFunction
from Himmelblau import Himmelblau
from GoldsteinPrice import GoldsteinPrice
from MakeContourPlot import MakeContourPlot
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import numpy as np
import random


plt.close('all')

function = lambda b: BeanFunction(b)
upper_bounds = [3,3]
lower_bounds = [-3,-1]
nVar = len(upper_bounds)

MakeContourPlot(function,upper_bounds,lower_bounds)

#plt.show()

# GA Parameters 
# crossover
ratio = 1.2 # scalar. If it lies in the range [0, 1], the children created are within the two parents. If algorithm is premature, try to set ratio larger than 1.0
crossfrac = 2/nVar #default in NSGA code
# mutation
scale = 0.1 #determines the standard deviation of the random numbers generated
shrink = 0.5 #scalar, [0,1]. As the optimization progress goes forward, decrease the mutation range (for example, shrink?[0.5, 1.0]) is usually used for local search
mutfrac = crossfrac # =2/nVar, default in NSGA code

# Stopping Criteria
maxgen = 50 # max number of generations

# Initialize Population NSGA uses random normally distributed points
#create first generation
pop = 15*nVar #should be 15-20 x number of design variables
#needs to be even
if (pop%2 == 1):
    pop+=1
# pop/2 should also be even
if ((pop/2)%2 == 1):
    pop+=2
    
# sample the solution space
engine = LatinHypercube(d=nVar)
points = np.array(engine.random(n=pop))
fitness = np.empty((pop,1))
# map each point value to the design space
i=0
for i, point in enumerate(points):
    # first shift to be centered at 0, then scale by design space size 
    # TODO: this should center the points on the bounds of the domain, rather than at 0
    point[0] = (point[0]-0.5)*(upper_bounds[0]-lower_bounds[0])
    point[1] = (point[1]-0.5)*(upper_bounds[1]-lower_bounds[1])    
    # evaluate the fitness of each point
    fitness[i] = function(point)
    
plt.scatter(points[:,0],points[:,1],c='red')
best_fitness = []
gen = 1
while gen <= maxgen:
    # determine best fitness and best point
    best = min(fitness)
    idx_best = np.where(fitness == best)
    best_point = points[idx_best]
    best_fitness.append(best)
    
    # determine the mating pool via tournament selection.
    # each point is randomly paired with another point, and the winner gets added to the mating pool
    pool = []
    num_tournaments = 0
    # do this twice
    while num_tournaments < 2:
        point_data = np.append(points,fitness,axis=1)
        while len(point_data)>=2:
        #for j in range(pop+1):
            opponents = random.sample(range(len(point_data)),2)
            if point_data[opponents[0],2] < point_data[opponents[1],2]:
                pool.append(point_data[opponents[0],0:2])
            else:
                pool.append(point_data[opponents[1],0:2])
            
            # get rid of these individuals from point_data
            point_data = np.delete(point_data,opponents,0)   
        num_tournaments += 1
    
    # generate offspring
    children = np.empty((0,2))
    fitness_children = np.empty((pop,1))
    # crossover
    # create parents
    #mom = []
    #dad = []
    while len(pool)>=2:
        mom = pool[0]
        dad = pool[1]
        crossover_flag = [random.random() < crossfrac for n in range(nVar)]
        randNum = [random.random() for n in range(nVar)]
        #child1 = np.empty((1,2))
        #child2 = np.empty((1,2))
        child1 = []
        child2 = []
        
        for j in range(nVar):
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
        del pool[0:2]
        
    # Mutate
    # calculate the mutation parameters using scale and shrink
    S = [scale*(1-shrink*gen/maxgen)*(upper_bounds[n]-lower_bounds[n]) for n in range(nVar)]
    # do the mutation
    for i, child in enumerate(children):
        for j in range(nVar):
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
    new_points = np.empty((0,nVar))
    new_fitness = np.empty((0,1))
    while len(new_points) < pop:
        best = min(all_fitness)
        idx_best = np.where(all_fitness == best)
        best_point = parents_with_children[idx_best]
        new_points = np.vstack((new_points,parents_with_children[idx_best]))
        new_fitness = np.vstack((new_fitness,all_fitness[idx_best]))
        parents_with_children = np.delete(parents_with_children,idx_best,0)   
        all_fitness = np.delete(all_fitness,idx_best,0)        
    fitness = new_fitness
    points = new_points
        
    # increment generation count
    gen+=1

    MakeContourPlot(function,upper_bounds,lower_bounds)
    plt.scatter(points[:,0],points[:,1],c='red')

# print best point and function value