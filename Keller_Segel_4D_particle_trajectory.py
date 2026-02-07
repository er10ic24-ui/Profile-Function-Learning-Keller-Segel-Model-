#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:53:40 2025

@author: chenqian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:46:39 2024

@author: chenqian
"""

import numpy as np 
import scipy 
import scipy.linalg as sl
import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection # For multicolored lines plots 
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import warnings 

# if you want a plot window, then type: %matplotlib qt at 
# at the command line 

# Solving the 3D Keller Segel model with merging algorithm by linked list data structure. 
d = 4 # dimension of the problem 
# Initial setting:  

n_particles = 50  # initial number of particles 
tau = 1e-4        # time step for the evolution
observed_time_step = 0.01  # we record the particle trajectories with time step: 0.005 

outer_iter = 2001 # number of iterations for the nonlinear optimization problem  
Stopping_time = np.around(tau * outer_iter, decimals = 2)  
Final_time_step = int(observed_time_step / tau)
t = np.linspace(0, Stopping_time, outer_iter)
Evolve_x = []
n_particle_curr = []

# BandWidth for Gaussian Kernel 
h = 1e-2  
# cut-off distance 
r_c = 0.05

r_c_square = r_c ** 2 
r_c_fourth = r_c ** 4
inverse_fourth_r_c = 1 / r_c_fourth

# initial mass for the density function 
omega = 1.0


# initial positions of the particles: depends on the initial density distribution  
# number of different initial positions 
number_of_initials = 1
BIG_X =[]; BIG_Y = []; BIG_Z = []; BIG_W = []   # Now we have two components 
x_final = []
BIG_X_dot = []; BIG_Y_dot = []; BIG_Z_dot = []; BIG_W_dot = []

for m in range(number_of_initials):
    print(m) 
    vec = np.random.uniform(low = 0.0, high = 1.0, size = [n_particles, 4])
    x_init = vec[:,0, None]
    y_init = vec[:,1, None]
    z_init = vec[:,2, None]
    w_init = vec[:,3, None]
    x = np.copy(x_init)
    y = np.copy(y_init)
    z = np.copy(z_init)
    w = np.copy(w_init) 
    
    
    # V = 0; W(x) = -1 / (4 * pi * |x|); H(x) = xlog(x) 
    grad_old_x = np.zeros(np.shape(x_init)); grad_old_y = np.zeros(np.shape(y_init)) 
    grad_old_z = np.zeros(np.shape(z_init)); grad_old_w = np.zeros(np.shape(w_init)) 
    x_old = np.zeros(np.shape(x_init)); y_old = np.zeros(np.shape(y_init)) 
    z_old = np.zeros(np.shape(z_init)); w_old = np.zeros(np.shape(w_init))
    X_old = []; Y_old = []; Z_old = []; W_old = [] 
    X_old.append(x); Y_old.append(y); Z_old.append(z); W_old.append(w)  
    X_dot = []; Y_dot = []; Z_dot = []; W_dot = []  
    print(m)
    
    for i in range(outer_iter):
        
    
        for j in range(100): # maximum number of iterations for BB method (or other optimization method)
            
            # Applying BB method for optimization 
            
            # For the dissipation term: 
            dist_term_x = (x - x_init) / tau 
            dist_term_y = (y - y_init) / tau 
            dist_term_z = (z - z_init) / tau 
            dist_term_w = (w - w_init) / tau 
            
            # To compute the gradient of the Keller-Segel kernel and the Gaussian Kernel 
            diff_x = x - np.transpose(x) # An N by N matrix consisting of x_{i} - x_{j}  
            diff_y = y - np.transpose(y) # An N by N matrix consisting of y_{i} - y_{j} 
            diff_z = z - np.transpose(z) # An N by N matrix consisting of z_{i} - z_{j} 
            diff_w = w - np.transpose(w) # An N by N matrix consisting of w_{i} - w_{j} 
            
           
            # For the KS_kernel  
            diff_sum_square = np.power(diff_x, 2) + np.power(diff_y, 2) + np.power(diff_z, 2) + np.power(diff_w, 2)   
            
            cut_off = np.piecewise(diff_sum_square, [diff_sum_square <= r_c_square, diff_sum_square > r_c_square ], [inverse_fourth_r_c, lambda y: 1 / (y ** 2)])
            
            # Gradient of Keller_Segel kernel 
              
            Keller_cut_off = omega / (2 * (np.pi ** 2)) * cut_off
            KS_cut_off_kernel_grad_x = diff_x * Keller_cut_off
            KS_cut_off_kernel_grad_x = np.sum(KS_cut_off_kernel_grad_x[:,:,None], axis = 1)
            
            KS_cut_off_kernel_grad_y = diff_y * Keller_cut_off
            KS_cut_off_kernel_grad_y = np.sum(KS_cut_off_kernel_grad_y[:,:,None], axis = 1)
            
            KS_cut_off_kernel_grad_z = diff_z * Keller_cut_off 
            KS_cut_off_kernel_grad_z = np.sum(KS_cut_off_kernel_grad_z[:,:,None], axis = 1)
            
            KS_cut_off_kernel_grad_w = diff_w * Keller_cut_off
            KS_cut_off_kernel_grad_w = np.sum(KS_cut_off_kernel_grad_w[:,:,None], axis = 1)
             
            
            # Gradient of the cut-off 2D Keller_Segel kernel
            
            np.fill_diagonal(diff_sum_square, 0)
            
            # Gaussian Kernel terms without constant factor 
            K_h = np.exp(- diff_sum_square / (h ** 2))
            
            sum_K_h = np.sum(K_h, axis = 1) # An N by 1 column vector 
            
            first_moment_x = diff_x * K_h
            first_moment_y = diff_y * K_h
            first_moment_z = diff_z * K_h
            first_moment_w = diff_w * K_h
            
            # Now we have all the components of the gradient terms for our optimization problem J 
            # We can compute the gradient of J now 
            
            # First term of the gradient of nonlocal Keller approximation of rho * log(rho)  
            nonlocal_grad1_x = -2 / (h ** 2) \
                * np.sum((first_moment_x / sum_K_h)[:,:,None], axis = 1)     # N by 1 vector for x-axis 
            
            nonlocal_grad1_y = -2 / (h ** 2) \
                * np.sum((first_moment_y / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for y-axis
            
            nonlocal_grad1_z = -2 / (h ** 2) \
                * np.sum((first_moment_z / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for z-axis  
            
            nonlocal_grad1_w = -2 / (h ** 2) \
                * np.sum((first_moment_w / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for w_axis 
                
            # Second term of the gradient of nonlocal Keller approximation of rho * log(rho)
            nonlocal_grad2_x = -2 / (h ** 2) \
                * np.sum(first_moment_x[:,:,None], axis = 1) / sum_K_h[:, None] # N by 1 vector 
            
            nonlocal_grad2_y = -2 / (h ** 2) \
                * np.sum(first_moment_y[:,:,None], axis = 1) / sum_K_h[:, None]
            
            nonlocal_grad2_z = -2 / (h ** 2) \
                * np.sum(first_moment_z[:,:,None], axis = 1) / sum_K_h[:, None]    
            
            nonlocal_grad2_w = -2 / (h ** 2) \
                * np.sum(first_moment_w[:,:,None], axis = 1) / sum_K_h[:, None]
            
            
            # Normal gradient             
            grad_x = dist_term_x + nonlocal_grad1_x + nonlocal_grad2_x + KS_cut_off_kernel_grad_x / n_particles # total gradient for the optimization problem 
            
            grad_y = dist_term_y + nonlocal_grad1_y + nonlocal_grad2_y + KS_cut_off_kernel_grad_y / n_particles  
            
            grad_z = dist_term_z + nonlocal_grad1_z + nonlocal_grad2_z + KS_cut_off_kernel_grad_z / n_particles 
            
            grad_w = dist_term_w + nonlocal_grad1_w + nonlocal_grad2_w + KS_cut_off_kernel_grad_w / n_particles 
             
            
            
            # cut-off gradient                                                                         # An N by 1 vctor 
            # grad = dist_term + nonlocal_grad1 + nonlocal_grad2 + Keller_cut_off_grad 
            
            if np.sqrt(np.tensordot(grad_x, grad_x) + np.tensordot(grad_y, grad_y) + np.tensordot(grad_z, grad_z) + np.tensordot(grad_w, grad_w) ) < 1e-6:   # Stopping criterion of the inner loop
                print(j)
                break                 
            
            step_length_x = 1e-6
            step_length_y = 1e-6 
            step_length_z = 1e-6 
            step_length_w = 1e-6 
            
            # Compute the BB step length 
            if j > 0: 
                w_x = grad_x - grad_old_x
                w_y = grad_y - grad_old_y
                w_z = grad_z - grad_old_z 
                w_w = grad_w - grad_old_w 
                
                s_x = x - x_old 
                s_y = y - y_old
                s_z = z - z_old 
                s_w = w - w_old 
                
                step_length_x = np.tensordot(s_x, s_x) / np.tensordot(w_x, s_x)
                step_length_y = np.tensordot(s_y, s_y) / np.tensordot(w_y, s_y)
                step_length_z = np.tensordot(s_z, s_z) / np.tensordot(w_z, s_z) 
                step_length_w = np.tensordot(s_w, s_w) / np.tensordot(w_w, s_w)
                
            grad_old_x = grad_x 
            grad_old_y = grad_y 
            grad_old_z = grad_z
            grad_old_w = grad_w 
            
            x_old = x; y_old = y; z_old = z; w_old = w 
            x = x - step_length_x * grad_x 
            y = y - step_length_y * grad_y 
            z = z - step_length_z * grad_z
            w = w - step_length_w * grad_w 
            
        if (i > 0) and (i % Final_time_step) == 0: 
            X_old.append(x_old)
            Y_old.append(y_old) 
            Z_old.append(z_old)
            W_old.append(w_old)
            
            X_dot.append((x - x_init) / tau) 
            Y_dot.append((y - y_init) / tau)
            Z_dot.append((z - z_init) / tau)
            W_dot.append((w - w_init) / tau)
            
        x_init = x; y_init = y; z_init = z; w_init = w   
        
        
    BIG_X.append(X_old); BIG_Y.append(Y_old)
    BIG_Z.append(Z_old); BIG_W.append(W_old) 
    
    BIG_X_dot.append(X_dot); BIG_Y_dot.append(Y_dot)
    BIG_Z_dot.append(Z_dot); BIG_W_dot.append(W_dot) 
    
BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y)
BIG_Z = np.array(BIG_Z); BIG_W = np.array(BIG_W)

BIG_Data = np.concatenate([BIG_X, BIG_Y, BIG_Z, BIG_W], axis = 3)
BIG_X_dot = np.array(BIG_X_dot); BIG_Y_dot = np.array(BIG_Y_dot)
BIG_Z_dot = np.array(BIG_Z_dot); BIG_W_dot = np.array(BIG_W_dot)

BIG_Derivative = np.concatenate([BIG_X_dot, BIG_Y_dot, BIG_Z_dot, BIG_W_dot], axis = 3)

record = 0
if record == 1:
    with open('KS_4DParticle_true_kernel_50_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)
        np.save(f, h)
        np.save(f, tau)
        np.save(f, omega)
        np.save(f, observed_time_step)
        np.save(f, BIG_Derivative) 

        
        
                                                  
                                                              
        
        

        
        
        



