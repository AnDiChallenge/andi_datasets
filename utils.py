# Copyright 2019 by Gorka Munoz-Gil under the MIT license.
# This file is part of the Anomalous diffusion challenge (AnDi), and is 
# released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included in the repository containing the file 
# (github.com/gorkamunoz/ANDI)

import numpy as np    


def bm1D(T, D, deltaT = False):
    '''Creates a 1D Brownian motion trajectory'''
    if D < 0:
        raise ValueError('Only positive diffusion coefficients allowed.') 
    if not deltaT:               
        deltaT = 1
    return np.cumsum(np.sqrt(2*D*deltaT)*np.random.randn(int(T)))        
        

def regularize(positions, times, T):
    '''Regularizes a trajectory with irregular sampling times.
    Arguments:
        - positions (numpy.array): collections of positions
        - times (numpy.array): collections of sampling times
        - T (int): lenght of the output trajectory
        
        return: numpy.array''' 
    times = np.append(0, times)
    pos_r = np.zeros(T)
    for idx in range(len(times)-1):
        pos_r[int(times[idx]):int(times[idx+1])] = positions[idx]
    pos_r -= pos_r[0]
    return pos_r
        
def sample_sphere(N, R):
    '''Samples random number that lay in the surface of a 3D sphere centered in 
    zero and with radius R
    Arguments:
        - N (int): number of points to generate.
        - R (int or numpy.array): radius of the sphere. If int, all points have
        the same radius, if numpy.array, each number has different radius.'''
    vecs = np.random.randn(3, N)
    vecs /= np.linalg.norm(vecs, axis=0)
    return R*vecs

def normalize(traj, dimensions):
    ''' Normalizes trajectories by substracting average and dividing by sqrt of msd
    Arguments:
	- traj: trajectory or ensemble of trajectories to normalize. 
    - dimensions: dimension of the trajectory.
	return: normalized trajectory'''
    # If there is a single trajectory to normalize, reshape to not have 
    # problems afterwards
    if len(traj.shape) == 1:
        traj = np.reshape(traj, (1, traj.shape[0])) 
    num_traj = traj.shape[0]
    len_traj = traj.shape[1]  
    variance = np.sqrt((traj**2).mean(axis=1, keepdims=True))
    variance[variance == 0] = 1 # If the variance is equal to zero, we put
                				# variance to one to do nothing in the next
				                # step.    
    norm_traj = (traj - traj.mean(axis=1, keepdims=True))/variance   
    norm_traj = norm_traj.reshape(dimensions*num_traj, int(len_traj/dimensions))
    return (norm_traj.transpose() - norm_traj[:,0]).transpose().reshape(num_traj, len_traj)
