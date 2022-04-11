# This script loads data from a DNS simulation of 2D viscous flow past two colinear 
# plates, aligned perpendicular to the freestream velocity
# 
# This data is stored on a uniformly-spaced 2D grid. The velocity components 
# are each saved in their own 2D array, where the first index is time, and the
# second is the spatial location. 
#
# Gap between plates = 1
#
# Reynolds number = 100 (based on freesteam velocity and the length of one 
# plate)
#
# Scott Dawson 3/13/2021

import numpy as np
import matplotlib.pyplot as plt
import h5py
Data = h5py.File('Data2PlatesGap1Re100_Alpha-5.mat','r')

# %% Load data and specify relevant parameters 
X = Data['DataX']; # x coordinates
Y = Data['DataY']; # y coordinates

U = Data['DataU']; # streamwise velocity field
V = Data['DataV']; # spanwise velocity field

nx = 1199; # gridpoints in x-direction
ny = 349; # gridpoints in y-direction
nt = 1000; # timesteps
dt = 1; # 1 convective time (based on a single plate length) between snapshots
dx = 0.08; # spatial resolution in x-direction
dy = 0.08; # spatial resolution in y-direction


# %% Reshape vectors for plotting purposes

Xgrid = np.reshape(X,(nx,ny),order='F')
Ygrid = np.reshape(Y,(nx,ny),order='F')

# %% Plot sample velocity field
snapInd = 100; # pick arbitrary timestep
Ugrid = np.reshape(U[snapInd,:],(nx,ny),order='F');
Vgrid = np.reshape(V[snapInd,:],(nx,ny),order='F');

nlevels = 20;
clevelsU =np.linspace(-0.5,2.5,nlevels);
clevelsV =np.linspace(-1,1,nlevels);

fig, axs = plt.subplots(2,1);
axs[0].contourf(Xgrid,Ygrid,Ugrid, clevelsU, cmap="viridis");
axs[0].plot([0,0],[-1.5,-0.5],'k'); # body 1
axs[0].plot([0,0],[0.5,1.5],'k'); # body 2
axs[0].axis('equal');
axs[0].set_xlabel('x');
axs[0].set_ylabel('y');
axs[0].set_title('Streamwise velocity');

axs[1].contourf(Xgrid,Ygrid,Vgrid, clevelsV, cmap="viridis");
axs[1].plot([0,0],[-1.5,-0.5],'k'); # body 1
axs[1].plot([0,0],[0.5,1.5],'k') ;# body 2
axs[1].axis('equal');
axs[1].set_xlabel('x');
axs[1].set_ylabel('y');
axs[1].set_title('Transverse velocity');
