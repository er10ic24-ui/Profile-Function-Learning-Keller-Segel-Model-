#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 15:39:44 2026

@author: chenqian
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib.collections import LineCollection # For multicolored lines plots 
import patsy
# Generate 2D Stochastic particle trajectories for the Keller Segel model 

# Initial settings: 
N_par = 50 # numebr of initial particles 
dt = 1e-4 # time increment 
T = 0.2 # Final time 
N_time = int(T / dt)
t = np.linspace(0, T, N_time + 1) 
observed_time_step = 1e-3 
record_time_step = int(observed_time_step / dt) 



omega = 4.0  # parameter value 
epsilon = 1e-2 # Regularization parameter 
epsilon_sqr = epsilon ** 2 
# Stochastic terms
# Brownian increments: dW ~ N(0, dt)
eta = 0.01 
sigma = eta * np.sqrt(2) # diffusion coefficient 

# To save the particle trajectory data 
number_of_initials = 20
BIG_X =[]; BIG_Y = []   # Now we have two components 
x_final = []
BIG_X_noise = []; BIG_Y_noise = [] # Store the noises  
BIG_Noise = []

# For each inital distribution 
for m in range(number_of_initials): 
    print(m) 
    X_old = []; Y_old = [] 
    X_noise = []; Y_noise = []
    X_init = np.random.uniform(0, 1, size = (N_par, 2))
    X_0 = X_init[:,0,None]; Y_0 = X_init[:,1,None]
    X_old.append(X_0); Y_old.append(Y_0)
    
    for i in range(N_time): 
        Diff_X = X_0 - X_0.T; Diff_Y = Y_0 - Y_0.T
        dist_diff = np.power(Diff_X, 2) + np.power(Diff_Y, 2) 

        x_noise = np.random.normal(0, 1, size = (N_par, 1)); y_noise = np.random.normal(0, 1, size = (N_par, 1)) 
        X_noise.append(x_noise); Y_noise.append(y_noise)
        
        X = X_0 + sigma * np.sqrt(dt) * x_noise + omega / N_par * (- np.sum(Diff_X / (2 * np.pi) / (dist_diff + epsilon_sqr), axis = 1, keepdims = True) ) * dt  
        Y = Y_0 + sigma * np.sqrt(dt) * y_noise + omega / N_par * (- np.sum(Diff_Y / (2 * np.pi) / (dist_diff + epsilon_sqr), axis = 1, keepdims = True) ) * dt    
        X_0 = np.copy(X); Y_0 = np.copy(Y) 

        
        if ((i + 1) % record_time_step) == 0: 
            X_old.append(X_0); Y_old.append(Y_0) 

    BIG_X.append(X_old); BIG_Y.append(Y_old)
    BIG_X_noise.append(X_noise); BIG_Y_noise.append(Y_noise) 


BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y)
BIG_X_noise = np.array(BIG_X_noise); BIG_Y_noise = np.array(BIG_Y_noise) 
BIG_Data = np.concatenate([BIG_X, BIG_Y], axis = 3)
BIG_Noise = np.concatenate([BIG_X_noise, BIG_Y_noise], axis = 3) 

a_min = []; b_max = []
    
for s in range(np.shape(BIG_X)[0]): # 2-norm 
        W_x = np.reshape(BIG_X[s], [np.shape(BIG_X)[1], np.shape(BIG_X)[2]])
        W_y = np.reshape(BIG_Y[s], [np.shape(BIG_Y)[1], np.shape(BIG_Y)[2]])
        square_pairwise_diff_x = np.power(np.diff(W_x), 2)
        square_pairwise_diff_y = np.power(np.diff(W_y), 2)
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y)
        
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
# Use the data-defined interval to do function approximation 
a_min = np.min(a_min); b_max = np.max(b_max)
nodes = np.linspace(a_min, b_max, 200)




record = 0
if record == 1:
    # with open('KS_2D_SDE_Par_' + str(N_par) + '_reg_' + str(epsilon) + '_t_' + str(T) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    with open('KS_2D_SDE_Par_' + str(N_par) + '_eta_' + str(eta) + '_reg_' + str(epsilon) + '_observed_time_step_' + str(observed_time_step) + '_t_' + str(T) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_2D_SDE_Par_50_deter_reg_' + str(epsilon) + '_t_' + str(T) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_2DParticle_true_kernel_50_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)
        # np.save(f, BIG_X)
        # np.save(f, BIG_Y)
        np.save(f, sigma) # store the noise level for the stochastic particle trajectories 
        np.save(f, epsilon) # store the regularization parameter  
        np.save(f, observed_time_step) # store the time step for the trajectory data
        np.save(f, BIG_Noise) 
        # np.save(f, dt) # store the time step for the particle evolution 
        # np.save(f, BIG_Derivative) 

X_old = np.array(X_old); Y_old = np.array(Y_old)

observed_time = np.arange(0, T + observed_time_step , observed_time_step)
num_plots = N_par 

plot = 1
if plot == 1: 
    fig, ax = plt.subplots() 
    # ax = plt.axes(xlim = (-1., 1.), ylim = (-1., 1.))
    # cbaxes = fig.add_axies([0.15, 0.03, 0.7])
    # cbaxes = fig.add_axies([0.15, 0.1, 0.03, 0.7])
    norm = plt.Normalize(observed_time[0], observed_time[-1]) 
    
    colormap = matplotlib.colormaps["plasma_r"] 
    
    for k in range(N_par): 
         
        xy = np.array([X_old[:,k,0], Y_old[:,k,0]]).T 
        segments = np.stack([xy[:-1, :], xy[1:, :]], axis = 1)
        
        lc = LineCollection(segments, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)
        ax.add_collection(lc)
        
        if k == 0:
            lc.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
            fig.colorbar(lc, ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
            ax.grid()
    
    ax.set_xlim(0,1) 
    ax.set_ylim(0,1)
    ax.set_xlabel("x_axis")
    ax.set_ylabel("y_axis")
    ax.set_title("Particle trajectory 2d with \u03C9 =" + str(omega))
    plt.show()



