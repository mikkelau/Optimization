# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:49:48 2022

@author: mikke
"""
import matplotlib.pyplot as plt 
from random import random
import math
from numpy.linalg import norm


def myfunc(x):
    f = x**3-6*x**2+12*x-8
    g = 3*x**2-12*x+12
    return f,g

def myKeplar(x):
    f = x-0.7*math.sin(x)-math.pi/2
    g = 1-0.7*math.cos(x)
    return f,g

f = 1
guess = random()*20-10
resids = []
print('starting guess:', guess)
while (norm(f) > 1e-14):
    f,g = myKeplar(guess)
    guess = guess-f/g
    resids.append(norm(f))

print('solution:',guess)
plt.plot(resids)

# naming the x axis 
plt.xlabel('Iterations') 
# naming the y axis 
plt.ylabel('Residual') 

plt.yscale("log")
  
# function to show the plot 
plt.show() 