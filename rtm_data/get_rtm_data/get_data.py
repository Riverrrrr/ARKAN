'''
Copyright (c) 2026 Beihang University

Developed by:
He Zichuan

Project:
ARKAN: Autoregressive Kolmogorov-Arnold Network for RTM Wavefield Prediction


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:


The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

from __future__ import division
import numpy as np
from bruges.filters.wavelets import ricker
import matplotlib.pyplot as plt
from scipy.signal import convolve
import skimage.filters
import pyekfmm as fmm
#import skimage.io

dt = 1e-4 #seconds
dx = 1    #meters
dz = 1    #meters

#Size of model in meters
nx = 256
nz = 256

tmax = 0.1 #seconds
t_array = np.arange(0, tmax, dt) #array of times

vp = 2000.0     #m/s
vmodel = vp * np.ones((nx, nz), dtype = np.float32)


# z1 = int(50/80*256)
# z2 = int(28/80*256)
# z3 = int(31/80*256)
# z4 = int(19/80*256)
# z5 = int(22/80*256)
# x1 = int(70/100*256)
# x2 = int(73/100*256)
# x3 = int(28/100*256)
# x4 = int(31/100*256)
# vmodel[:, z1:] = vp*1.3
# vmodel[x1:x2, z2:z3] = vp*1.4
# vmodel[x3:x4, z4:z5] = vp*1.6


vmodel[:, 50:] = vp*1.3
vmodel[70:73, 28:31] = vp*1.4
vmodel[28:31, 19:22] = vp*1.6

wav1 = ricker(duration = 0.012, dt = dt, f = 200)

isx = 50
isz = 0

u_all = np.zeros((np.shape(vmodel)[0], np.shape(vmodel)[1], int(tmax/dt)), dtype = float)

r = vmodel*dt/dx

def solve_fd2d(v, w, vmodel, r, dt, dx, dz):
    """
    Compute wave amplitude at the next k-th time step
    v = snapshot of amplitude at last step (k-1)
    w = snapshot of amplitude at previous step (k-2).
    """
    
    D = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    
    return 2*v - 1*w + (r**2)*convolve(v, D, mode='same')


for k, tk in enumerate(t_array):
    if k >= 2: # we need to start from step 2 because step 1 and step 0 are initial conditions
        v = u_all[:,:,k-1]
        w = u_all[:,:,k-2]
        
        u = solve_fd2d(v, w, vmodel, r, dt, dx, dz)

        if k < len(wav1.amplitude): # if source is active, add its amplitudes to the wavefield
            u[isx, isz] = u[isx, isz] + wav1.amplitude[k]

        u_all[:,:,k] = u


print(u_all.shape)


def solve_fd2d_abc(v, w, vmodel, r, dt, dx, dz):
    """
    Compute wave amplitude at the next k-th time step
    with boundary conditions
    
    v = snapshot of amplitude at last step (k-1)
    w = snapshot of amplitude at previous step (k-2).
    """
    
    D = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    u_out = 2*v - 1*w + (r**2)*convolve(v, D, mode='same')
    
    #bottom
    u_out[:,-1] = v[:,-2] + (r[:,-1] - 1)/(r[:,-1] + 1)*(u_out[:,-2] - v[:,-1])
    #right
    u_out[-1,:] = v[-2,:] + (r[-1,:] - 1)/(r[-1,:] + 1)*(u_out[-2,:] - v[-1,:])
    #left
    u_out[0,:] = v[1,:] + (r[0,:] - 1)/(r[0,:] + 1)*(u_out[1,:] - v[0,:])
    
    return u_out

u_abc = np.zeros((np.shape(vmodel)[0], np.shape(vmodel)[1], int(tmax/dt)), dtype = float)
for k, tk in enumerate(t_array):
    if k >= 2: # we need to start from step 2 because step 1 and step 0 are initial conditions
        v = u_abc[:,:,k-1]
        w = u_abc[:,:,k-2]
        
        u = solve_fd2d_abc(v, w, vmodel, r, dt, dx, dz)

        if k < len(wav1.amplitude): # if source is active, add its amplitudes to the wavefield
            u[isx, isz] = u[isx, isz] + wav1.amplitude[k]

        u_abc[:,:,k] = u


# surface_record_no_bc = u_all[:,1,:]
# surface_record_raw = u_abc[:,1,:]

# #display

# #Top muting
# muted_gather = surface_record_raw.copy()
# x_array = np.arange(0, nx*dx, dx)
# v0 = vmodel[:,0]
# traveltimes = abs(np.cumsum(dx/v0) - np.cumsum(dx/v0)[isx])
# for traceno in range(len(x_array)):
#     muted_gather[traceno, 0:int(traveltimes[traceno]/dt  + len(wav1.amplitude))] = 0 
    

np.save('u_abc.npy', u_abc)
np.save('vmodel.npy', vmodel)