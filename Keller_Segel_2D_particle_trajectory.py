#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:51:11 2024

@author: chenqian
"""

import numpy as np 
import scipy 
import scipy.linalg as sl
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection # For multicolored lines plots 
import matplotlib 
import warnings 

# if you want a plot window, then type: %matplotlib qt at 
# at the command line 

# Solving the 2D Keller Segel model with merging algorithm by linked list data structure. 
d = 2 # dimension of the problem 
# Initial setting:  

n_particles = 50  # initial number of particles 
tau = 1e-4        # time step for the evolution
observed_time_step = 0.01  # we record the particle trajectories with time step

outer_iter = 2001 # number of iterations for the nonlinear optimization problem  
Stopping_time = np.around(tau * outer_iter, decimals = 2)  
Final_time_step = int(observed_time_step / tau)
t = np.linspace(0, Stopping_time, outer_iter)
Evolve_x = []
n_particle_curr = []


# BandWidth for Gaussian Kernel 
h = 0.01 

# cut-off distance 
r_c = 0.01 

r_c_square = r_c ** 2 

inverse_square_r_c = 1 / r_c_square  

# initial weight for each particle: Need for merging algorithm 
weight = 1 / n_particles * np.ones([n_particles, 1] )

# initial mass for the density function 
omega = 2.0 

# initial positions of the particles: depends on the initial density distribution  
# number of different initial positions 
number_of_initials = 10
BIG_X =[]; BIG_Y = []   # Now we have two components 
x_final = []
BIG_X_dot = []; BIG_Y_dot = [] 

for m in range(number_of_initials):
    vec = np.random.uniform(low = 0.0, high = 1.0, size = [n_particles, 2])
    x_init = vec[:,0,None]
    y_init = vec[:,1,None]
    x = np.copy(x_init)
    y = np.copy(y_init)
    
    
    grad_old_x =  np.zeros(np.shape(x_init));    grad_old_y = np.zeros(np.shape(y_init))
    x_old = np.zeros(np.shape(x_init));     y_old = np.zeros(np.shape(y_init))
    X_old = []; Y_old = []
    X_old.append(x); Y_old.append(y) 
    X_dot = []; Y_dot = []
    print(m)
    
    for i in range(outer_iter-1):
        
    
        for j in range(100): # maximum number of iterations for BB method (or other optimization method)
            
            # Applying BB method for optimization 
            
            # For the dissipation term: 
            dist_term_x = (x - x_init) / tau 
            dist_term_y = (y - y_init) / tau 
            
            
            
            # To compute the gradient of the Keller-Segel kernel and the Gaussian Kernel 
            diff_x = x - np.transpose(x) # An N by N matrix consisting of x_{i} - x_{j}  
            diff_y = y - np.transpose(y) # An N by N matrix consisting of y_{i} - y_{j} 
            
           
            # For the KS_kernel  
            # np.fill_diagonal(diff_x, 1) # Warning: change the diagonal terms to be zero 
            # np.fill_diagonal(diff_y, 1) 
            
            diff_sum_square = np.power(diff_x, 2) + np.power(diff_y, 2)  
            
            cut_off = np.piecewise(diff_sum_square, [diff_sum_square <= r_c_square , diff_sum_square > r_c_square], [inverse_square_r_c, lambda y: 1 / y])
            
            # Gradient of Keller_Segel kernel 
              
            Keller_cut_off = omega / (2 * np.pi) * cut_off
            KS_cut_off_kernel_grad_x = diff_x * Keller_cut_off
            KS_cut_off_kernel_grad_x = np.sum(KS_cut_off_kernel_grad_x[:,:,None], axis = 1)
            
            KS_cut_off_kernel_grad_y = diff_y * Keller_cut_off
            KS_cut_off_kernel_grad_y = np.sum(KS_cut_off_kernel_grad_y[:,:,None], axis = 1)
            
           
            np.fill_diagonal(diff_sum_square, 0)
            
            # Gaussian Kernel terms without constant factor 
            K_h = np.exp(- diff_sum_square / (h ** 2))
            
            sum_K_h = np.sum(K_h, axis = 1) # An N by 1 column vector 
            
            
            first_moment_x = diff_x * K_h
            first_moment_y = diff_y * K_h
            
            
            # Now we have all the components of the gradient terms for our optimization problem J 
            # We can compute the gradient of J now 
            
            # First term of the gradient of nonlocal Keller approximation of rho * log(rho)  
            nonlocal_grad1_x = -2 / (h ** 2) \
            * np.sum((first_moment_x / sum_K_h)[:,:,None], axis = 1)     # N by 1 vector for x-axis 
            
            nonlocal_grad1_y = -2 / (h ** 2) \
                * np.sum((first_moment_y / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for y-axis 
            
            # Second term of the gradient of nonlocal Keller approximation of rho * log(rho)
            nonlocal_grad2_x = -2 / (h ** 2) \
            * np.sum(first_moment_x[:,:,None], axis = 1) / sum_K_h[:, None] # N by 1 vector 
            
            
            nonlocal_grad2_y = -2 / (h ** 2) \
            * np.sum(first_moment_y[:,:,None], axis = 1) / sum_K_h[:, None]
            
            
            
            
            # Normal gradient 
            grad_x = dist_term_x + nonlocal_grad1_x + nonlocal_grad2_x + KS_cut_off_kernel_grad_x / n_particles # total gradient for the optimization problem 
            
            grad_y = dist_term_y + nonlocal_grad1_y + nonlocal_grad2_y + KS_cut_off_kernel_grad_y / n_particles  
            
            
            if np.sqrt(np.tensordot(grad_x, grad_x) + np.tensordot(grad_y, grad_y)) < 1e-6:   # Stopping criterion of the inner loop
                print(j)
                break                 
            
            step_length_x = 1e-5
            step_length_y = 1e-5 
            # Compute the BB step length 
            if j > 0: 
                z_x = grad_x - grad_old_x
                z_y = grad_y - grad_old_y
                s_x = x - x_old 
                s_y = y - y_old
                step_length_x = np.tensordot(s_x, s_x) / np.tensordot(s_x, z_x)
                step_length_y = np.tensordot(s_y, s_y) / np.tensordot(s_y, z_y)
            
            grad_old_x = grad_x 
            grad_old_y = grad_y 
            x_old = x; y_old = y  
            x = x - step_length_x * grad_x 
            y = y - step_length_y * grad_y 
        
        if (i > 0) and (i % Final_time_step) == 0: 
            X_old.append(x_old)
            Y_old.append(y_old) 
            X_dot.append((x - x_init) / tau); Y_dot.append((y - y_init) / tau)
            
        x_init = x; y_init = y 
        
        
    BIG_X.append(X_old); BIG_Y.append(Y_old)
    BIG_X_dot.append(X_dot); BIG_Y_dot.append(Y_dot)
    
BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y)
BIG_Data = np.concatenate([BIG_X, BIG_Y], axis = 3)
BIG_X_dot = np.array(BIG_X_dot); BIG_Y_dot = np.array(BIG_Y_dot)
BIG_Derivative = np.concatenate([BIG_X_dot, BIG_Y_dot], axis = 3)


record = 0
if record == 1:
    with open('KS_2D_Particle_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)
        np.save(f, h)
        np.save(f, tau)
        np.save(f, observed_time_step)
        np.save(f, BIG_Derivative) 

# To plot particle trajectory in multicolored lines 


observed_time = np.arange(0,Stopping_time + observed_time_step , observed_time_step)
num_plots = n_particles 



fig, ax = plt.subplots() 
# ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
# cbaxes = fig.add_axies([0.15, 0.03, 0.7])
# cbaxes = fig.add_axies([0.15, 0.1, 0.03, 0.7])
norm = plt.Normalize(observed_time[0], observed_time[-1]) 

colormap = matplotlib.colormaps["plasma_r"] 

for k in range(n_particles): 
     
    xy = np.array([BIG_X[0][:,k,0], BIG_Y[0][:,k,0]]).T 
    segments = np.stack([xy[:-1, :], xy[1:, :]], axis = 1)
    
    lc = LineCollection(segments, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)
    ax.add_collection(lc)
    # ax.add_patch(lc)
    
    if k == 0:
        # sm = fig.cm.ScalarMappable(cmap = colormap)
        # sm.set_clim(vmin = 0.0, vmax = 0.2)
        lc.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
        # fig.colorbar(lc, cax = cbaxes, ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
        fig.colorbar(lc, ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
        ax.grid()

ax.set_xlabel("x_axis")
ax.set_ylabel("y_axis")
ax.set_title("Particle trajectory 2d with \u03C9 =" + str(omega))
plt.show()








        
                                                  
                                                              
        
        
        
        
        



