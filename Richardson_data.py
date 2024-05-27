import h5py
import os
import numpy as np


print( "Begin reading of HDF5 file: full_traj_tracers.h5 ")

N = 100000 # Number of trajectories to read

xfile = h5py.File('/home/tau/apantea/data/full_traj_tracers.h5', 'r')    
traj = xfile['traj3d'][0:N,:,0:3] # read dataset from particle 0 to particle N
vel = xfile['traj3d'][0:N,:,3:6]
xfile.close()

print(traj.shape, vel.shape)


np.save("/home/tau/apantea/data/trajectories_R_3d.npy", traj)
np.save("/home/tau/apantea/data/velocities_R_3d.npy", vel)

