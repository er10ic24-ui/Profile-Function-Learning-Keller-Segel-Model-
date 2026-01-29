#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:44:00 2024

@author: chenqian
"""


import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist  
import patsy # for comparison
from scipy.interpolate import PchipInterpolator


def data_distribution(a,b, BIG_Data, bins_number = 400):
    # [a, b] is the domain of the data distribution 
    # bins_numebr represents the number of subintervals in [a, b]
    # BIG_Data is the total data of the particle trajectories with different time and different initial data
    # BIG_Data should be of the form: M(num of initials) * L(num of recorded times) * N(num of particles) * d(dim of each particle)
    
    bins =  np.linspace(a, b, bins_number)
    interval_length = bins[1] - bins[0] # use uniform subdivision 

    bins_midpoint = (bins[:-1] + bins[1:]) / 2 # (bins_number - 1) of midpoints  

    Count_of_bins = np.zeros(np.size(bins) - 1) # collect the number od data point fell into each bin 

    for s in range(np.shape(BIG_Data)[0]):
        for l in range(np.shape(BIG_Data)[1]):
            
            pair_dist = pdist(BIG_Data[s][l])      
            Count_of_bins += np.histogram(pair_dist, bins)[0]
    
    discrete_density = Count_of_bins / np.sum(Count_of_bins) / interval_length 
    

    # interp_density = interp1d(bins_midpoint, discrete_density, fill_value = 'extrapolate')
    interp_density = PchipInterpolator(bins_midpoint, discrete_density)

    return interp_density, bins, discrete_density 

    
    