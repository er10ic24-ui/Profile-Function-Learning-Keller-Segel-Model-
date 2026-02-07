# Keller–Segel Particle Trajectories: Deterministic and Stochastic Learning
A unifed learning of the profile function of the Keller-Segel model 

## Overview 
This repository contains research code and results for studying particle-based formulations of the Keller–Segel model in multiple spatial dimensions.
We generate and analyze deterministic and stochastic particle trajectories in two, three, and four dimensions, and use these trajectories to learn interaction (profile) functions governing the particle dynamics.

Due to storage constraints, only a subset of generated trajectory data is included, while all figures corresponding to the reported results are provided.

## Model Background

The Keller–Segel model describes collective behavior driven by aggregation and diffusion mechanisms.
In this project, the model is studied through a particle system representation, allowing both deterministic and stochastic dynamics to be examined across dimensions.

## Contents of This Repository

### 1. Particle Trajectory Generation
	•	Deterministic particle trajectories
	•	2D, 3D, and 4D
	•	Stochastic particle trajectories
	•	2D and 4D

⚠️ Data availability note:
Due to file size limitations, only deterministic particle trajectories in 1D, 2D, and 3D are included in the repository.
Higher-dimensional and stochastic trajectory data were used to generate figures but are not uploaded.

### 2. Learning of Profile Functions
	•	Learned interaction/profile functions for:
	•	Deterministic particle systems
	•	Stochastic particle systems
	•	Results reported for all considered dimensions

These learned profiles characterize the effective interaction structure inferred from particle trajectories.


### Particle trajectories for the Keller–Segel model
<p align="center">
  <img src="figures/particle_trajectory_2d.png" width="500">
</p>
Figure 1: 2D particle trajectories colored by time.
