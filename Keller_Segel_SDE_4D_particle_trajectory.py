#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 15:45:00 2026

@author: chenqian
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib.collections import LineCollection # For multicolored lines plots 
import KS_data_distribution as KS
import patsy
# Generate 4D Stochastic particle trajectories for the Keller Segel model 

# Initial settings: 
N_par = 50 # numebr of initial particles 
dt = 1e-4 # time increment 
T = 0.2 # Final time 
N_time = int(T / dt)
t = np.linspace(0, T, N_time + 1) 
observed_time_step = 1e-3 
record_time_step = int(observed_time_step / dt) 
d = 4 


omega = 4.0  # parameter value 
epsilon = 1e-2 # Regularization parameter 
epsilon_sqr = epsilon ** 2 
# Stochastic terms
# Brownian increments: dW ~ N(0, dt)
eta = 0.01 
sigma = eta * np.sqrt(2) # diffusion coefficient 

# To save the particle trajectory data 
number_of_initials = 1
BIG_X =[]; BIG_Y = []; BIG_Z = []; BIG_W = []   # Now we have four components 
x_final = []
BIG_X_noise = []; BIG_Y_noise = []; BIG_Z_noise = []; BIG_W_noise = [] # Store the noises  
BIG_Noise = []

# For each inital distribution 
for m in range(number_of_initials): 
    print(m) 
    X_old = []; Y_old = []; Z_old = []; W_old = []  
    X_noise = []; Y_noise = []; Z_noise = []; W_noise = [] 
    X_init = np.random.uniform(0, 1, size = (N_par, d))
    X_0 = X_init[:,0,None]; Y_0 = X_init[:,1,None]; Z_0 = X_init[:,2,None]; W_0 = X_init[:,3,None] 
    X_old.append(X_0); Y_old.append(Y_0); Z_old.append(Z_0); W_old.append(W_0) 
    
    for i in range(N_time): 
        Diff_X = X_0 - X_0.T; Diff_Y = Y_0 - Y_0.T; Diff_Z = Z_0 - Z_0.T; Diff_W = W_0 - W_0.T 
        dist_diff = np.power(Diff_X, 2) + np.power(Diff_Y, 2) + np.power(Diff_Z, 2) + np.power(Diff_W, 2) 

        x_noise = np.random.normal(0, 1, size = (N_par, 1)); y_noise = np.random.normal(0, 1, size = (N_par, 1))
        z_noise = np.random.normal(0, 1, size = (N_par, 1)); w_noise = np.random.normal(0, 1, size = (N_par, 1)) 
        X_noise.append(x_noise); Y_noise.append(y_noise); Z_noise.append(z_noise); W_noise.append(w_noise) 
        
        X = X_0 + sigma * np.sqrt(dt) * x_noise + omega / N_par * (-np.sum(Diff_X / (2 * np.pi ** 2) / ((dist_diff ** 2) + epsilon_sqr), axis = 1, keepdims = True) ) * dt  
        Y = Y_0 + sigma * np.sqrt(dt) * y_noise + omega / N_par * (-np.sum(Diff_Y / (2 * np.pi ** 2) / ((dist_diff ** 2) + epsilon_sqr), axis = 1, keepdims = True) ) * dt    
        Z = Z_0 + sigma * np.sqrt(dt) * z_noise + omega / N_par * (-np.sum(Diff_Z / (2 * np.pi ** 2) / ((dist_diff ** 2) + epsilon_sqr), axis = 1, keepdims = True) ) * dt 
        W = W_0 + sigma * np.sqrt(dt) * w_noise + omega / N_par * (-np.sum(Diff_W / (2 * np.pi ** 2) / ((dist_diff ** 2) + epsilon_sqr), axis = 1, keepdims = True) ) * dt 
        X_0 = np.copy(X); Y_0 = np.copy(Y); Z_0 = np.copy(Z); W_0 = np.copy(W)  

        
        if ((i + 1) % record_time_step) == 0: 
            X_old.append(X_0); Y_old.append(Y_0); Z_old.append(Z_0); W_old.append(W_0)  

    BIG_X.append(X_old); BIG_Y.append(Y_old); BIG_Z.append(Z_old); BIG_W.append(W_old)
    BIG_X_noise.append(X_noise); BIG_Y_noise.append(Y_noise); BIG_Z_noise.append(Z_noise); BIG_W_noise.append(W_noise)  


BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y); BIG_Z = np.array(BIG_Z); BIG_W = np.array(BIG_W)
BIG_X_noise = np.array(BIG_X_noise); BIG_Y_noise = np.array(BIG_Y_noise); BIG_Z_noise = np.array(BIG_Z_noise); BIG_W_noise = np.array(BIG_W_noise) 
BIG_Data = np.concatenate([BIG_X, BIG_Y, BIG_Z, BIG_W], axis = 3)
BIG_Noise = np.concatenate([BIG_X_noise, BIG_Y_noise, BIG_Z_noise, BIG_W_noise], axis = 3) 

a_min = []; b_max = []
    
for s in range(np.shape(BIG_X)[0]): # 2-norm 
        W_x = np.reshape(BIG_X[s], [np.shape(BIG_X)[1], np.shape(BIG_X)[2]])
        W_y = np.reshape(BIG_Y[s], [np.shape(BIG_Y)[1], np.shape(BIG_Y)[2]])
        W_z = np.reshape(BIG_Z[s], [np.shape(BIG_Z)[1], np.shape(BIG_Z)[2]]) 
        W_w = np.reshape(BIG_W[s], [np.shape(BIG_W)[1], np.shape(BIG_W)[2]]) 
        square_pairwise_diff_x = np.power(np.diff(W_x), 2)
        square_pairwise_diff_y = np.power(np.diff(W_y), 2)
        square_pairwise_diff_z = np.power(np.diff(W_z), 2)
        square_pairwise_diff_w = np.power(np.diff(W_w), 2)
        
        pairwise_distance = np.sqrt(square_pairwise_diff_x + square_pairwise_diff_y \
                                    + square_pairwise_diff_z + square_pairwise_diff_w)
        
        a_min.append(np.min(pairwise_distance)); b_max.append(np.max(pairwise_distance))
        
# Use the data-defined interval to do function approximation 
a_min = np.min(a_min); b_max = np.max(b_max)
nodes = np.linspace(a_min, b_max, 200)

record = 0
if record == 1:
    # with open('KS_2D_SDE_Par_' + str(N_par) + '_reg_' + str(epsilon) + '_t_' + str(T) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    with open('KS_4D_SDE_Par_' + str(N_par) + '_eta_' + str(eta) + '_reg_' + str(epsilon) + '_observed_time_step_' + str(observed_time_step) + '_t_' + str(T) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_2D_SDE_Par_50_deter_reg_' + str(epsilon) + '_t_' + str(T) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
    # with open('KS_2DParticle_true_kernel_50_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)
        np.save(f, sigma) # store the noise level for the stochastic particle trajectories 
        np.save(f, epsilon) # store the regularization parameter  
        np.save(f, observed_time_step) # store the time step for the trajectory data
        np.save(f, BIG_Noise) 
        

fig1, ax1,  = plt.subplots()
density, bins, discrete_density = KS.data_distribution(a_min, b_max, BIG_Data)
interp_density = density(nodes)
# ax1.plot(nodes, W, label = 'Cut off kernel', markersize = 7, color = 'black', linestyle = 'solid')
# ax1.plot(nodes, Learned_Kernel, label = 'Learned kernel', markersize = 5, color = 'b', linestyle = '--', dashes = (5,5) )
ax1.plot(nodes, interp_density, color = [0.85, 0.325, 0.098], alpha = 0.2)
ax1.fill_between(nodes, interp_density, color = [0.85, 0.325, 0.098], alpha = 0.2, zorder = 1)




