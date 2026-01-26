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
d = 3 # dimension of the problem 
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
h = 0.01 

# cut-off distance 
r_c = 0.01

r_c_square = r_c ** 2 
r_c_cube = r_c ** 3
inverse_cubic_r_c = 1 / r_c_cube  


# initial weight for each particle: Need for merging algorithm 
weight = 1 / n_particles * np.ones([n_particles, 1] )

# initial mass for the density function 
omega = 2.0

# initial positions of the particles: depends on the initial density distribution  
# number of different initial positions 
number_of_initials = 1
BIG_X =[]; BIG_Y = []; BIG_Z = []   # Now we have two components 
x_final = []
BIG_X_dot = []; BIG_Y_dot = []; BIG_Z_dot = []

for m in range(number_of_initials):
    print(m) 
    vec = np.random.uniform(low = 0.0, high = 1.0, size = [n_particles, 3])
    x_init = vec[:,0, None]
    y_init = vec[:,1, None]
    z_init = vec[:,2, None]
    x = np.copy(x_init)
    y = np.copy(y_init)
    z = np.copy(z_init)
    
    
    # V = 0; W(x) = -1 / (4 * pi * |x|); H(x) = xlog(x) 
    grad_old_x =  np.zeros(np.shape(x_init)); grad_old_y = np.zeros(np.shape(y_init)); grad_old_z = np.zeros(np.shape(z_init))
    x_old = np.zeros(np.shape(x_init)); y_old = np.zeros(np.shape(y_init)); z_old = np.zeros(np.shape(z_init))
    X_old = []; Y_old = []; Z_old = []
    X_old.append(x); Y_old.append(y); Z_old.append(z) 
    X_dot = []; Y_dot = []; Z_dot = [] 
    
    
    for i in range(outer_iter):
        
    
        for j in range(100): # maximum number of iterations for BB method (or other optimization method)
            
            # Applying BB method for optimization 
            
            # For the dissipation term: 
            dist_term_x = (x - x_init) / tau 
            dist_term_y = (y - y_init) / tau 
            dist_term_z = (z - z_init) / tau 
            
            
            # To compute the gradient of the Keller-Segel kernel and the Gaussian Kernel 
            diff_x = x - np.transpose(x) # An N by N matrix consisting of x_{i} - x_{j}  
            diff_y = y - np.transpose(y) # An N by N matrix consisting of y_{i} - y_{j} 
            diff_z = z - np.transpose(z) # An N by N matrix consisting of z_{i} - z_{j} 
            
           
            # For the KS_kernel  
            diff_sum_square = np.power(diff_x, 2) + np.power(diff_y, 2) + np.power(diff_z, 2)  
                        
            cut_off = np.piecewise(diff_sum_square, [diff_sum_square <= r_c_square, diff_sum_square > r_c_square ], [inverse_cubic_r_c, lambda y: 1 / (y ** 1.5)])

            # Gradient of Keller_Segel kernel 
            Keller_cut_off = omega / (4 * np.pi) * cut_off
            KS_cut_off_kernel_grad_x = diff_x * Keller_cut_off
            KS_cut_off_kernel_grad_x = np.sum(KS_cut_off_kernel_grad_x[:,:,None], axis = 1)
            
            KS_cut_off_kernel_grad_y = diff_y * Keller_cut_off
            KS_cut_off_kernel_grad_y = np.sum(KS_cut_off_kernel_grad_y[:,:,None], axis = 1)
            
            KS_cut_off_kernel_grad_z = diff_z * Keller_cut_off 
            KS_cut_off_kernel_grad_z = np.sum(KS_cut_off_kernel_grad_z[:,:,None], axis = 1)
            
            np.fill_diagonal(diff_sum_square, 0)
            
            # Gaussian Kernel terms without constant factor 
            K_h = np.exp(- diff_sum_square / (h ** 2))
            
            sum_K_h = np.sum(K_h, axis = 1) # An N by 1 column vector 
            
            first_moment_x = diff_x * K_h
            first_moment_y = diff_y * K_h
            first_moment_z = diff_z * K_h        
            
            # Now we have all the components of the gradient terms for our optimization problem J 
            # We can compute the gradient of J now 
            
            # First term of the gradient of nonlocal Keller approximation of rho * log(rho)  
            nonlocal_grad1_x = -2 / (h ** 2) \
                * np.sum((first_moment_x / sum_K_h)[:,:,None], axis = 1)     # N by 1 vector for x-axis 
            
            nonlocal_grad1_y = -2 / (h ** 2) \
                * np.sum((first_moment_y / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for y-axis
            
            nonlocal_grad1_z = -2 / (h ** 2) \
                * np.sum((first_moment_z / sum_K_h)[:,:,None], axis = 1) # N by 1 vector for z-axis  
            
            # Second term of the gradient of nonlocal Keller approximation of rho * log(rho)
            nonlocal_grad2_x = -2 / (h ** 2) \
                * np.sum(first_moment_x[:,:,None], axis = 1) / sum_K_h[:, None] # N by 1 vector 
            
            
            nonlocal_grad2_y = -2 / (h ** 2) \
                * np.sum(first_moment_y[:,:,None], axis = 1) / sum_K_h[:, None]
            
            nonlocal_grad2_z = -2 / (h ** 2) \
                * np.sum(first_moment_z[:,:,None], axis = 1) / sum_K_h[:, None]    
            
            
            # Normal gradient 
            # grad_x = dist_term_x + nonlocal_grad1_x + nonlocal_grad2_x + KS_kernel_grad_x / n_particles # total gradient for the optimization problem 
            grad_x = dist_term_x + nonlocal_grad1_x + nonlocal_grad2_x + KS_cut_off_kernel_grad_x / n_particles # total gradient for the optimization problem 
            
            # grad_y = dist_term_y + nonlocal_grad1_y + nonlocal_grad2_y + KS_kernel_grad_y / n_particles  
            grad_y = dist_term_y + nonlocal_grad1_y + nonlocal_grad2_y + KS_cut_off_kernel_grad_y / n_particles  
            
            grad_z = dist_term_z + nonlocal_grad1_z + nonlocal_grad2_z + KS_cut_off_kernel_grad_z / n_particles 
            # cut-off gradient                                                                         # An N by 1 vctor 
            # grad = dist_term + nonlocal_grad1 + nonlocal_grad2 + Keller_cut_off_grad 
            
            if np.sqrt(np.tensordot(grad_x, grad_x) + np.tensordot(grad_y, grad_y) + np.tensordot(grad_z, grad_z)) < 1e-4:   # Stopping criterion of the inner loop
                print(j)
                break                 
            
            step_length_x = 1e-6
            step_length_y = 1e-6 
            step_length_z = 1e-6 
            # Compute the BB step length 
            if j > 0: 
                w_x = grad_x - grad_old_x
                w_y = grad_y - grad_old_y
                w_z = grad_z - grad_old_z 
                s_x = x - x_old 
                s_y = y - y_old
                s_z = z - z_old 
                step_length_x = np.tensordot(s_x, s_x) / np.tensordot(w_x, s_x)
                step_length_y = np.tensordot(s_y, s_y) / np.tensordot(w_y, s_y)
                step_length_z = np.tensordot(s_z, s_z) / np.tensordot(w_z, s_z) 
            
            grad_old_x = grad_x 
            grad_old_y = grad_y 
            grad_old_z = grad_z 
            x_old = x; y_old = y; z_old = z   
            x = x - step_length_x * grad_x 
            y = y - step_length_y * grad_y 
            z = z - step_length_z * grad_z 
            
        if (i > 0) and (i % Final_time_step) == 0: 
            X_old.append(x_old)
            Y_old.append(y_old) 
            Z_old.append(z_old)
            X_dot.append((x - x_init) / tau) 
            Y_dot.append((y - y_init) / tau)
            Z_dot.append((z - z_init) / tau)

        x_init = x; y_init = y; z_init = z  
        
        
    BIG_X.append(X_old); BIG_Y.append(Y_old); BIG_Z.append(Z_old)
    BIG_X_dot.append(X_dot); BIG_Y_dot.append(Y_dot); BIG_Z_dot.append(Z_dot)
    
BIG_X = np.array(BIG_X); BIG_Y = np.array(BIG_Y); BIG_Z = np.array(BIG_Z)
BIG_Data = np.concatenate([BIG_X, BIG_Y, BIG_Z], axis = 3)
BIG_X_dot = np.array(BIG_X_dot); BIG_Y_dot = np.array(BIG_Y_dot); BIG_Z_dot = np.array(BIG_Z_dot)
BIG_Derivative = np.concatenate([BIG_X_dot, BIG_Y_dot, BIG_Z_dot], axis = 3)

record = 0
if record == 1:
    with open('KS_3D_Particle_50_cut_off_' + str(r_c) + '_t_' + str(Stopping_time) + '_Omega_' + str(omega) + '_'  + str(number_of_initials) + '.npy', 'wb') as f:
        np.save(f, BIG_Data)
        np.save(f, h)
        np.save(f, tau)
        np.save(f, observed_time_step)
        np.save(f, BIG_Derivative) 


observed_time = np.arange(0, Stopping_time + observed_time_step , observed_time_step)
colormap = matplotlib.colormaps["plasma_r"]

norm = plt.Normalize(observed_time[0], observed_time[-1])

fig = plt.figure()
ax = plt.axes(projection = '3d')
cbaxes = fig.add_axes([0.15, 0.1, 0.03, 0.7])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
   
# plt.colorbar(im, cax=cax) 

for k in range(n_particles): 
     
    xyz = np.array([BIG_X[0][:,k,0], BIG_Y[0][:,k,0], BIG_Z[0][:,k,0]]).T 
    segments = np.stack([xyz[:-1, :], xyz[1:, :]], axis = 1)
    lc = Line3DCollection(segments, linewidths = 2, colors = colormap(norm(observed_time)), cmap = colormap)
    ax.add_collection3d(lc)
    
    if k == 0:
        lc.set_clim(vmin = observed_time[0], vmax = observed_time[-1])
        fig.colorbar(lc, cax = cbaxes, ticks = [0.0, 0.05, 0.1, 0.15, 0.2]) # , fraction = 0.046, pad = 0.04)
    
    
    
ax.set_xlabel("x_axis")
ax.set_ylabel("y_axis")
ax.set_zlabel("z_axis")     
ax.set_title("Particle trajectory 3d with \u03C9 =" + str(omega))
plt.show()

        
                                                  
                                                              
        
        
        
        
        



