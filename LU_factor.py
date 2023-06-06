# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 20:08:30 2022

@author: mikke
"""
import numpy as np

def LU_factor(b,A):
    n = len(b)
    y = np.empty((n,))
    u = np.empty((n,))
    L = np.identity(n)
    U = np.array(A,dtype=np.float64) # need to specify floats so that nothing gets truncated to ints
    # want to go row by row, then column by column, then calculate column by column
    for row in range(1,n): # don't do the first row
        for column in range(row): # don't turn the diagonal into 0's, so make column=row the top of the exclusive range
            # find the multiplier
            mult = float(U[row][column]/U[column][column])
            # modify L appropriately
            L[row][column] = mult
            for j in range(n): # iterate through all columns of that row
                U[row][j] = float(U[row][j]-mult*U[column][j])
        
    y[0] = b[0]/L[0][0]
    for i in range(1,n):
        big_sum = 0
        for j in range(i):
            big_sum += L[i][j]*y[j] 
        yi = (1/L[i][i])*(b[i]-big_sum)
        y[i] = yi
        
    u[n-1] = y[n-1]/U[n-1][n-1]
    for i in reversed(range(n-1)):
        big_sum = 0
        for j in range(i+1,n):
            big_sum += U[i][j]*u[j]
        u[i] = (1/U[i][i])*(y[i]-big_sum)
    return u