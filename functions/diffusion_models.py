# Copyright 2019 by Gorka Munoz-Gil under the MIT license.
# This file is part of the Anomalous diffusion challenge (AnDi), and is 
# released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included in the repository containing the file 
# (github.com/gorkamunoz/ANDI)

''' 
This file contains a recopilatory of diffusion models used in ANDI. The class
is organized in three subclasses, each corresponding to a different dimensionality
of the output trajectory.

Currently the library containts the following models:
    Function   Dimensions    Description
    - bm          (1D)       Brownian motion
    - fbm      (1D/2D/3D)    Fractional browian motion, simulated by the fbm python library
    - ctrw     (1D/2D/3D)    Continuous time random walks
    - lw       (1D/2D/3D)    Levy walks
    - attm     (1D/2D/3D)    Annealed transit time
    - sbm      (1D/2D/3D)    Scaled brownian motion
        
Inputs of generator functions:
    - T (int): lenght of the trajectory. Gets transformed to int if input
                is float.
    - alpha (float): anomalous exponent
    
    Some generator functions also have optional inputs, see each function for
    details.
                            
Outputs:
    - numpy.array of lenght d.T, where d is the dimension
    
    Some generator functions have optional outputs, see each function for details
'''

import numpy as np
import fbm
from .utils_andi import regularize, bm1D, sample_sphere
from math import pi as pi
from scipy.special import erfcinv


__all__ = ['diffusion_models']

class diffusion_models(object):
        
    def __init__(self):
        '''Constructor of the class''' 
            
    class oneD():
        '''Class cointaning one dimensional diffusion models'''                   
            
        def fbm(self, T, alpha):
            ''' Creates a 1D fractional brownian motion trajectory'''            
            H = alpha*0.5
            return fbm.fbm(int(T-1), H)
        
        def ctrw(self, T, alpha, regular_time = True):
            ''' Creates a 1D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time. ''' 
            if alpha > 1:
                raise ValueError('Continuous random walks only allow for anomalous exponents <= 1.') 
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1-np.random.rand(T))**(-1/alpha))      
            times = times[:np.argmax(times>T)+1]
            # Generate the positions of the walk
            positions = np.cumsum(np.random.randn(len(times)))
            positions -= positions[0]           
            # Output
            if regular_time:
                return regularize(positions, times, T)
            else:
                return np.stack((times, positions))
            
        def lw(self, T, alpha):
            ''' Creates a 1D Levy walk trajectory '''             
            if alpha < 1:
                raise ValueError('Levy walks only allow for anomalous exponents > 1.')
            # Define exponents for the distribution of flight times                            
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3-alpha
            dt = (1-np.random.rand(T))**(-1/sigma)
            dt[dt > T] = T+1
            # Define the velocity
            v = 10*np.random.rand()                        
            # Generate the trajectory
            positions = np.empty(0)
            for t in dt:
                positions = np.append(positions, v*np.ones(int(t))*(2*np.random.randint(0,2)-1))
                if len(positions) > T:
                    break 
            return np.cumsum(positions[:int(T)]) - positions[0]
        
        def attm(self, T, alpha, regime = 1):   
            '''Creates a 1D trajectory following the annealed transient time model
            Optional parameters:
                :regime (int):
                    - Defines the ATTM regime. Accepts three values: 0,1,2.'''
            if regime not in [0,1,2]:
                raise ValueError('ATTM has only three regimes: 0, 1 or 2.')
            if alpha > 1:
                raise ValueError('ATTM only allows for anomalous exponents <= 1.') 
            # Gamma and sigma selection
            if regime == 0:
                sigma = 3*np.random.rand()
                gamma = np.random.uniform(low = -5, high = sigma)
                if alpha < 1:
                    raise ValueError('ATTM regime 0 only allows for anomalous exponents = 1.') 
            elif regime == 1:
                sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
                gamma = sigma/alpha
                while sigma > gamma or gamma > sigma + 1:
                    sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
                    gamma = sigma/alpha
            elif regime == 2:
                gamma = 1/(1-alpha)
                sigma = np.random.uniform(low = 1e-2, high = gamma-1)
            # Generate the trajectory  
            positions = np.array([0])
            while len(positions) < T:
                Ds =(1-np.random.uniform(low=0, high=1))**(1/sigma)  
                # Solving overflow problems
                if Ds == 0: 
                    Ds = 0; ts = T
                else:
                    ts = Ds**(-gamma)
                # Avoiding too long bm1D
                if ts > T:
                    ts = T
                # Calculate displacements
                dists = np.sqrt(2*Ds)*np.random.randn(int(ts))
                positions = np.append(positions, dists)
            positions = np.cumsum(positions)
            return positions[:T]-positions[0]
        
        
        def sbm(self, T, alpha, sigma = 1):            
            '''Creates a scaled brownian motion trajectory'''
            msd = (sigma**2)*np.arange(T+1)**alpha
            dx = np.sqrt(msd[1:]-msd[:-1])
            dx = np.sqrt(2)*dx*erfcinv(2-2*np.random.rand(len(dx)))
            return np.cumsum(dx)-dx[0]
        
    
    class twoD(oneD):  
            
        def ctrw(self, T, alpha, regular_time = True):
            ''' Creates a 2D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time. ''' 
            if alpha > 1:
                raise ValueError('Continuous random walks only allow for anomalous exponents <= 1.') 
            # Generate the waiting times from power-law distribution            
            times = np.cumsum((1-np.random.rand(T))**(-1/alpha))   
            times = times[:np.argmax(times>T)+1]
            # Generate the positions of the walk            
            posX = np.cumsum(np.random.randn(len(times)))
            posY = np.cumsum(np.random.randn(len(times)))            
            posX -= posX[0] 
            posY -= posY[0] 
            # Regularize and output            
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)                
                return np.concatenate((regX, regY))
            else:
                return np.stack((times, posX, posY))
            
        def fbm(self, T, alpha):
            ''' Creates a 2D fractional brownian motion trajectory'''
            # Defin Hurst exponent
            H = alpha*0.5            
            return np.concatenate((fbm.fbm(int(T-1), H), fbm.fbm(int(T-1), H)))
        
            
        def lw(self, T, alpha):
            ''' Creates a 2D Levy walk trajectory '''             
            if alpha < 1:
                raise ValueError('Levy walks only allow for anomalous exponents > 1.')             
            # Define exponents for the distribution of times              
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3-alpha
            dt = (1-np.random.rand(T))**(-1/sigma)
            dt[dt > T] = T+1
            # Define the velocity
            v = 10*np.random.rand()                        
            # Define the array where we save step length
            d= np.empty(0)
            # Define the array where we save the angle of the step
            angles = np.empty(0)
            # Generate trajectory
            for t in dt:
                d = np.append(d, v*np.ones(int(t))*(2*np.random.randint(0,2)-1))                
                angles = np.append(angles, np.random.uniform(low = 0, high = 2*pi)*np.ones(int(t)))
                if len(d) > T:
                    break
            d = d[:int(T)]  
            angles = angles[:int(T)] 
            posX, posY = [d*np.cos(angles), d*np.sin(angles)]         
            return np.concatenate((np.cumsum(posX)-posX[0], np.cumsum(posY)-posY[0]))
        
        
        def attm(self, T, alpha, regime = 1):   
            '''Creates a 2D trajectory following the annealed transient time model
            Optional parameters:
                :regime (int):
                    - Defines the ATTM regime. Accepts three values: 0,1,2.'''
            if regime not in [0,1,2]:
                raise ValueError('ATTM has only three regimes: 0, 1 or 2.')
            if alpha > 1:
                raise ValueError('ATTM only allows for anomalous exponents <= 1.')                 
            # Gamma and sigma selection
            if regime == 0:
                sigma = 3*np.random.rand()
                gamma = np.random.uniform(low = -5, high = sigma)
                if alpha < 1:
                    raise ValueError('ATTM regime 0 only allows for anomalous exponents = 1.')  
            elif regime == 1:
                sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
                gamma = sigma/alpha
                while sigma > gamma or gamma > sigma + 1:
                    sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
                    gamma = sigma/alpha
            elif regime == 2:
                gamma = 1/(1-alpha)
                sigma = np.random.uniform(low = 1e-2, high = gamma-1)
            # Generate the trajectory  
            posX = np.array([0])
            posY = np.array([0])            
            while len(posX) < T:
                Ds =(1-np.random.uniform(low=0, high=1))**(1/sigma)  
                # Solving overflow problems
                if Ds == 0: 
                    Ds = 0; ts = T
                else:
                    ts = Ds**(-gamma)
                # Avoiding too long bm1D
                if ts > T:
                    ts = T    
                # Generate displacements:
                distX = np.sqrt(2*Ds)*np.random.randn(int(ts))
                distY = np.sqrt(2*Ds)*np.random.randn(int(ts))                
                posX = np.append(posX, distX)
                posY = np.append(posY, distY)
                
            posX, posY = np.cumsum(posX), np.cumsum(posY)            
            return np.concatenate((posX[:T], posY[:T]))
        
        
        def sbm(self, T, alpha, sigma = 1):            
            '''Creates a scaled brownian motion trajectory'''
            msd = (sigma**2)*np.arange(T+1)**alpha
            deltas = np.sqrt(msd[1:]-msd[:-1])
            dx = np.sqrt(2)*deltas*erfcinv(2-2*np.random.rand(len(deltas)))            
            dy = np.sqrt(2)*deltas*erfcinv(2-2*np.random.rand(len(deltas)))  
            return np.concatenate((np.cumsum(dx)-dx[0], np.cumsum(dy)-dy[0]))
           
    class threeD(oneD):
            
        def ctrw(self, T, alpha, regular_time = True):
            ''' Creates a 3D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time. ''' 
            if alpha > 1:
                raise ValueError('Continuous random walks only allow for anomalous exponents <= 1.') 
            # Generate the waiting times from power-law distribution            
            times = np.cumsum((1-np.random.rand(T))**(-1/alpha))
            times = np.append(0, times)        
            times = times[:np.argmax(times>T)+1]
            # Generate the positions of the walk
            lengths = np.random.randn(len(times))
            posX, posY, posZ = np.cumsum(sample_sphere(len(times), lengths), axis=1)
            posX = posX - posX[0]
            posY = posY - posY[0]
            posZ = posZ - posZ[0]    
            # Regularize and output            
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T) 
                regZ = regularize(posZ, times, T)  
                return np.concatenate((regX, regY, regZ))
            else:
                return np.stack((times, posX, posY, posZ))
            
        def fbm(self, T, alpha): 
            ''' Creates a 3D fractional brownian motion trajectory'''
            # Define Hurst exponent
            H = alpha*0.5
            return np.concatenate((fbm.fbm(int(T-1), H), fbm.fbm(int(T-1), H), fbm.fbm(int(T-1), H)))
        
        
        def lw(self, T, alpha, regular_time = True):
            ''' Creates a 3D Levy walk trajectory '''             
            if alpha < 1:
                raise ValueError('Levy walks only allow for anomalous exponents > 1.')       
            # Define exponents for the distribution of times
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3-alpha
            dt = (1-np.random.rand(T))**(-1/sigma)
            dt[dt>T] = T+1
            # Define the velocity
            v = 10*np.random.rand()                        
            # Create the trajectory
            posX = np.empty(0)
            posY = np.empty(0)
            posZ = np.empty(0)
            for t in dt:
                distX, distY, distZ = sample_sphere(1, v)
                posX = np.append(posX, distX*np.ones(int(t)))
                posY = np.append(posY, distY*np.ones(int(t)))
                posZ = np.append(posZ, distZ*np.ones(int(t)))
                if len(posX) > T:
                    break
            return np.concatenate((np.cumsum(posX[:T])-posX[0], 
                                   np.cumsum(posY[:T])-posY[0],
                                   np.cumsum(posZ[:T])-posZ[0]))
        
            
        def attm(self, T, alpha, regime = 1):   
            '''Creates a 3D trajectory following the annealed transient time model
            Optional parameters:
                :regime (int):
                    - Defines the ATTM regime. Accepts three values: 0,1,2.'''
            if regime not in [0,1,2]:
                raise ValueError('ATTM has only three regimes: 0, 1 or 2.')
            if alpha > 1:
                raise ValueError('ATTM only allows for anomalous exponents <= 1.') 
            # Parameter selection
            if regime == 0:
                sigma = 3*np.random.rand()
                gamma = np.random.uniform(low = -5, high = sigma)
                if alpha < 1:
                    raise ValueError('ATTM Regime 0 can only produce trajectories with anomalous exponents = 1')
            elif regime == 1:
                sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
                gamma = sigma/alpha
                while sigma > gamma or gamma > sigma + 1:
                    sigma = 3*np.random.uniform(low = 1e-2, high = 1.1)
                    gamma = sigma/alpha
            elif regime == 2:
                gamma = 1/(1-alpha)
                sigma = np.random.uniform(low = 1e-2, high = gamma-1)
            # Create the trajectory  
            posX = np.array([0])
            posY = np.array([0])   
            posZ = np.array([0])
            while len(posX) < T:
                Ds =(1-np.random.uniform(low=0, high=1))**(1/sigma)  
                # Solving overflow problems
                if Ds == 0: 
                    Ds = 0; ts = T
                else:
                    ts = Ds**(-gamma)
                # Avoiding too long bm1D
                if ts > T:
                    ts = T 
                steps = np.sqrt(2*Ds)*np.random.randn(int(ts))
                distX, distY, distZ = sample_sphere(len(steps), steps)
                # Append the displacement
                posX = np.append(posX, distX)
                posY = np.append(posY, distY)
                posZ = np.append(posZ, distZ)
            # Do final cumsum    
            posX, posY, posZ = np.cumsum(posX), np.cumsum(posY), np.cumsum(posZ)
                
            return np.concatenate((posX[:T], posY[:T], posZ[:T]))
        
        
        def sbm(self, T, alpha, sigma = 1):            
            '''Creates a scaled brownian motion trajectory'''
            msd = (sigma**2)*np.arange(T+1)**alpha
            deltas = np.sqrt(msd[1:]-msd[:-1])
            dx = np.sqrt(2)*deltas*erfcinv(2-2*np.random.rand(len(deltas)))            
            dy = np.sqrt(2)*deltas*erfcinv(2-2*np.random.rand(len(deltas))) 
            dz = np.sqrt(2)*deltas*erfcinv(2-2*np.random.rand(len(deltas))) 
            return np.concatenate((np.cumsum(dx)-dx[0], np.cumsum(dy)-dy[0], np.cumsum(dz)-dz[0]))
