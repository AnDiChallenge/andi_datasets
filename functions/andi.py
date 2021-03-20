# Copyright 2019 by Gorka Munoz-Gil under the MIT license.
# This file is part of the Anomalous diffusion challenge (AnDi), and is 
# released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included in the repository containing the file 
# (github.com/gorkamunoz/ANDI)

import numpy as np
import os
import inspect
import h5py
from tqdm import trange
import csv

from .utils_andi import normalize
from .diffusion_models import diffusion_models

__all__ = ['andi_datasets']


class andi_datasets():
     
    def __init__(self):        
        '''Constructor of the class'''
        self.available_models = inspect.getmembers(diffusion_models.oneD(), inspect.ismethod)              
        self.avail_models_name = [x[0] for x in self.available_models]
        
    def get_models(self):        
        '''Loading subclass of models'''
        if self.dimension == 1:
            self.models = diffusion_models.oneD()
        elif self.dimension == 2:
            self.models = diffusion_models.twoD()
        elif self.dimension == 3:
            self.models = diffusion_models.threeD()
        else:
            raise ValueError('Our current understanding of the physical world is three dimensional and so are the diffusion models available in this class')
                
        self.available_models = inspect.getmembers(self.models, inspect.ismethod)      
        self.avail_models_name = [x[0] for x in self.available_models]
        self.avail_models_func = [x[1] for x in self.available_models]

    @property
    def n_models(self): return len(self.avail_models_name)
    
    def create_dataset(self, T, N, exponents, models,
                       dimension = 1,
                       save_trajectories = False, load_trajectories = False, 
                       path = 'datasets/',
                       N_save = 1000, t_save = 1000):        
        ''' Create a dataset of trajectories
        Arguments:
            :T (int):
                - length of the trajectories.   
            :N (int, numpy.array):
                - if int, number of trajectories per class (i.e. exponent and model) in the dataset.
                - if numpy.array, number of trajectories per classes: size (number of models)x(number of classes)    
            :exponents (float, array):
                - anomalous exponents to include in the dataset. Allows for two digit precision.
            :models (bool, int, list):
                - labels of the models to include in the dataset. Correspodance between models and labels
                  is given by self.label_correspodance, defined at init.
                  If int/list, choose the given models. If False, choose all of them.
            :dimensions (int):
                - Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
            :save_trajectories (bool):
                - - if True, the module saves a .h5 file for each model considered, with N_save trajectories 
                  and T = T_save.
            :load_trajectories (bool):
                - if True, the module loads the trajectories of an .h5 file.
            :path (str):
                - path to the folder where to save/load the trajectories dataset.
            :N_save (int):
                - Number of trajectories to save for each exponents/model. Advise: save at the beggining
                  a big dataset (t_save ~ 1e3 and N_save ~ 1e4) which allows you to load any other combiantion
                  of T and N.
            :t_save (int):
                - Length of the trajectories to be saved. See comments on N_save.                
        Return:
            :data_models (numpy.array):
                - Dataset of trajectories of lenght Nx(T+2), with the following structure:
                    o First column: model label 
                    o Second column: value of the anomalous exponent
                    o 2:T columns: trajectories'''
                    
        '''Managing probable errors in inputs'''
        if T < 2:
            raise ValueError('The time of the trajectories has to be bigger than 1.')       
        if isinstance(exponents, int) or isinstance(exponents, float):
            exponents = [exponents]
        
        '''Managing folders of the datasets'''       
        if save_trajectories or load_trajectories:                
            if load_trajectories:
                save_trajectories = False            
            if not os.path.exists(path) and load_trajectories:
                raise FileNotFoundError('The directory from where you want to load the dataset does not exist')                
            if not os.path.exists(path) and save_trajectories:
                os.makedirs(path)  
                
        '''Establish dimensions and corresponding models'''
        self.dimension = dimension
        self.get_models()
                
        '''Managing models to load'''       
        # Load from a list of models
        if isinstance(models, list): 
            self.models_name = [self.avail_models_name[idx] for idx in models]     
            self.models_func = [self.avail_models_func[idx] for idx in models]
        # Load from a single model
        elif isinstance(models, int) and not isinstance(models, bool):
            self.models_name = [self.avail_models_name[models]]
            self.models_func = [self.avail_models_func[models]]
        # Load all available models
        else: 
            self.models_name =  self.avail_models_name
            self.models_func =  self.avail_models_func
            
        '''Managing number of trajectory per class:
            - Defines array num_class as a function of N'''                            
        if isinstance(N, int): 
            n_per_class = N*np.ones((len(self.models_name), len(exponents)))
            
        elif type(N).__module__ == np.__name__: 
            if len(self.models_name) != N.shape[0] or len(exponents) != N.shape[1]:
                raise ValueError('Mismatch between the dimensions of N and the number of different classes.'+
                                 f'N must be either an int (balanced classes) or an array of length {len(models)}x'
                                 f'{len(exponents)} (inbalaced classes).') 
            n_per_class = N
        else:
            raise TypeError('Type of variable N not recognized.')
                    
        '''Defining default values for saved datasets''' 
        N_save = np.ones_like(n_per_class)*N_save
        # If the number of class of a given class is bigger than N_save, we
        # change the value of N_save for that particular class.
        N_save = np.max([N_save, n_per_class], axis = 0)      
                
        
        ''' Loading/Saving/Creating datasets'''
        if load_trajectories:
            data_models = self.load_trajectories(T = T,
                                                 exponents = exponents,
                                                 models_name = self.models_name,
                                                 dimension = self.dimension,
                                                 n_per_class = n_per_class,
                                                 path = path,
                                                 N_save = N_save,
                                                 t_save = t_save)
        elif save_trajectories:
            self.save_trajectories(exponents = exponents,
                                   dimension = self.dimension,
                                   models_name = self.models_name,
                                   models_func = self.models_func,
                                   path = path, 
                                   n_per_class = n_per_class,
                                   N_save = N_save,
                                   t_save = t_save)
            
            data_models = self.load_trajectories(T = T,
                                                 exponents = exponents,
                                                 dimension = self.dimension,
                                                 models_name = self.models_name,                                                 
                                                 n_per_class = n_per_class,
                                                 path = path,
                                                 N_save = N_save,
                                                 t_save = t_save)
            
        else:           
            data_models = self.create_trajectories(T = T,                                                   
                                                   exponents = exponents, 
                                                   dimension = self.dimension,
                                                   models_name = self.models_name,
                                                   models_func = self.models_func,
                                                   n_per_class = n_per_class)       
            
        return data_models
    
    def load_trajectories(self, T, exponents, dimension, 
                                models_name, n_per_class, 
                                path, N_save = 1000, t_save = 1000):
        ''' Load trajectories from a h5py file of the given path. The name of the datasets in the
        file have the following structure: 
            '(exponent with 2 digit_precision)_T_(lenght of trajectories in the dataset)_N_(number of trajectories in the dataset)'
        Arguments: 
            :T (int):
                - length of the trajectories.   
            :exponents (array):
                - anomalous exponents to include in the dataset. Allows for two digit precision.
            :dimension (int):
                - Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
            :models_name (list of str):
                - names of the models to include in the output dataset. 
            :n_per_class:
                - number of trajectories to consider per exponent/model.
            :path (str):
                - path to the folder from where to load the trajectories dataset.
            :t_save (int):
                - length of the trajectories in the datasets to load.
            :N_save (array):
                - number of trajectories contained in the datasets to load.                  
        Return:
            :dataset (numpy.array):
                - Dataset of trajectories of lenght (number of models)x(T+2), with the following structure:
                    o First column: model label 
                    o Second column: value of the anomalous exponent
                    o 2:T columns: trajectories'''
                    
        '''Establish dimensions and corresponding models'''
        self.dimension = dimension
        self.get_models()
            
        
        if isinstance(models_name, int):
            models_name = [models_name]
               
        for idx_m, name in enumerate(models_name):        
            hf = h5py.File(path+name+'.h5', 'r+')
            
            for idx_e, exp  in enumerate(exponents):
                
                name_dataset = f'{exp:.2f}_T_{t_save}_N_'+ \
                                str(int(N_save[idx_m, idx_e]))+f'_dim_{self.dimension}'  
                
                n = int(n_per_class[idx_m, idx_e])
                if n == 0:
                    continue
                
                try:
                    data = (hf.get(name_dataset)[()][:n,:self.dimension*T]) 
                except:
                    raise TypeError('The dataset you want to load does not exist.')
                    
                
                data = self.label_trajectories(trajs = data, model_name = name, exponent = exp)                
                            
                if idx_e + idx_m == 0:
                    dataset = data
                else:
                    dataset = np.concatenate((dataset, data), axis = 0) 
        return dataset
     
    def save_trajectories(self, exponents, models_name, models_func, path, n_per_class,
                          N_save = 1000, t_save = 1000, dimension = 1):
        ''' Saves a dataset for the exponents and models considered. 
        Arguments:   
            :exponents (array):
                - anomalous exponents to include in the dataset. Allows for two digit precision.
            :models_name (list of str):
                - names of the models to include in the output dataset. 
            :models_func (list of funcs):
                - function generating the models to include in the output dataset. 
            :path (str):
                - path to the folder where to save the trajectories dataset.
            :t_save (int):
                - length of the trajectories to save in the datasets.
            :N_save (array):
                - number of trajectories to include in the datasets saved.
            :dimension (int):
                - Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
        No return           '''     
    
        '''Establish dimensions and corresponding models'''
        self.dimension = dimension
        self.get_models()        
        
        for idx_m, (name, func) in enumerate(zip(models_name, models_func)):
            
            if os.path.isfile(path+name+'.h5'):
                action = 'r+'
            else:
                action = 'w'
            with h5py.File(path+name+'.h5', action) as hf:
                
                for idx_e, exp in enumerate(exponents): 
                    if n_per_class[idx_m, idx_e] == 0:
                        continue
                    
                    n = int(N_save[idx_m, idx_e])                    
                    name_dataset = f'{exp:.2f}_T_{t_save}_N_{n}_dim_{self.dimension}' 
                    
                    if name_dataset not in hf:  
                        
                        data = np.zeros((n, self.dimension*t_save))                           
                        # TQDM variables
                        tq = trange(n)
                        tq.set_postfix(saving = True, model = name, exponent = exp)
                        for i in tq:
                            data[i, :] = func(t_save, exp)                           
                            
                        hf.create_dataset(name_dataset, data=data)
                        
                    else:
                        print(f'The dataset for {name} with exponent {round(exp,3)}'
                                +' already exists, no need of saving it again.')
            
        
    def create_trajectories(self, T, exponents, dimension, models_name, models_func, n_per_class):  
        ''' Saves a dataset for the exponents and models considered. 
        Arguments:  
            :T (int):
                - length of the trajectories.   
            :exponents (array):
                - anomalous exponents to include in the dataset. Allows for two digit precision.
            :dimension (int):
                - Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
            :models_name (list of str):
                - names of the models to include in the output dataset. 
            :models_func (list of funcs):
                - function generating the models to include in the output dataset. 
            :n_per_class:
                - number of trajectories to consider per exponent/model. 
        Return:
            :dataset (numpy.array):
                - Dataset of trajectories of lenght (number of models)x(T+2), with the following structure:
                    o First column: model label.
                    o Second column: value of the anomalous exponent.
                    o 2:T columns: trajectories.'''
            
        for idx_m, (name, func) in enumerate(zip(models_name, models_func)):
            for idx_e, exp in enumerate(exponents):
                
                
                n = int(n_per_class[idx_m, idx_e])
                data = np.zeros((n, self.dimension*T))  
                for i in range(n):
                    data[i, :] = func(T, exp)
                    
                data = self.label_trajectories(trajs = data, model_name = name, exponent = exp)   
                
                if idx_e + idx_m == 0:
                    dataset = data
                else:
                    dataset = np.concatenate((dataset, data), axis = 0)
                
        return dataset
                
            
    def label_trajectories(self, trajs, model_name, exponent):
        ''' Labels given trajectories given the corresponding label for the model and exponent.
        For models, the label correspond to the position of the model in self.avail_models_name.
        For exponents, the label if the value of the exponent.
        Arguments:
            :trajs (numpy array):
                - trajectories to label
            :model_name (str):
                - name of the model from which the trajectories are coming from.
            :exponent (float):
                - Anomalous exponent of the trajectories. 
        Return:
            :trajs (numpy array):
                - Labelled trajectoreis, with the following structure:
                    o First column: model label
                    o Second columnd: exponent label
                    o Rest of the array: trajectory.   '''
        
        label_model = self.avail_models_name.index(model_name)          
         
        labels_mod = np.ones((trajs.shape[0], 1))*label_model
        labels_alpha = np.ones((trajs.shape[0], 1))*exponent
        trajs = np.concatenate((labels_mod, labels_alpha, trajs), axis = 1)
        
        return trajs

    def create_noisy_localization_dataset(self, 
                                          dataset = False,
                                          T = False, N = False, exponents = False, models = False, dimension = 1,
                                          noise_func = False, sigma = 1, mu = 0,
                                          save_trajectories = False, load_trajectories = False, 
                                          path = 'datasets/',
                                          N_save = 1000, t_save = 1000): 
        ''' Create a dataset of noisy trajectories. This function creates trajectories with create_trajectories
        and then adds given noise to them.        
        Arguments: All arguments are the same as create_trajectories but noise_func
            :dataset (bool, numpy array):
                - If False, creates a dataset with the given parameters. If numpy array, dataset to which the
                  function applies the noise.
            :noise_func (bool, function):
                - if False, the noise added to the trajectories will be Gaussian distributed, with 
                  variance sigma and mean value mu.
                - if function, uses the given function to generate noise to be added to the trajectory. The 
                  function must have as input two ints, N and M and the output must be a matrix of size NxM.
        Return:
            :data_models (numpy.array):
                - Dataset of trajectories of lenght Nx(T+2), with the following structure:
                    o First column: model label 
                    o Second column: value of the anomalous exponent
                    o 2:T columns: trajectories'''
                    
        if not dataset.any():
            dataset = self.create_dataset(T, N, exponents, models, dimension,
                                                     save_trajectories, load_trajectories, 
                                                     path,
                                                     N_save, t_save)
            
        # Add the noise to the trajectories  
        trajs = dataset[:, 2:].reshape(dataset.shape[0]*dimension, T)
        trajs = self.add_noisy_localization(trajs, noise_func, sigma, mu)
        
        dataset[:, 2:] = trajs.reshape(dataset.shape[0], T*dimension)
        
        return dataset    
    
    def create_noisy_diffusion_dataset(self, 
                                       dataset = False,
                                       T = False, N = False, exponents = False, models = False, dimension = 1,
                                       diffusion_coefficients = False,
                                       save_trajectories = False, load_trajectories = False, 
                                       path = 'datasets/',
                                       N_save = 1000, t_save = 1000): 
        ''' Create a dataset of noisy trajectories. This function creates trajectories with `create_trajectories`
        and then adds given noise to them.        
        Arguments: All arguments are the same as `create_trajectories` but dataset and diffusion_coefficients
            :dataset (bool, numpy array):
                - If False, creates a dataset with the given parameters. If numpy array, dataset to which the
                  function applies the noise.
            :noise_func (bool, function):
                - if False, the noise added to the trajectories will be Gaussian distributed, with 
                  variance sigma and mean value mu.
                - if function, uses the given function to generate noise to be added to the trajectory. The 
                  function must have as input two ints, N and M and the output must be a matrix of size NxM.
                 - if numpy array, sums it to the trajectories
        Return:
            :data_models (numpy.array):
                - Dataset of trajectories of lenght Nx(T+2), with the following structure:
                    o First column: model label 
                    o Second column: value of the anomalous exponent
                    o 2:T columns: trajectories'''
                    
        if not dataset.any():
            dataset = self.create_dataset(T, N, exponents, models, dimension,
                                                     save_trajectories, load_trajectories, 
                                                     path,
                                                     N_save, t_save)
        # Add the noise to the trajectories 
        trajs = dataset[:, 2:].reshape(dataset.shape[0]*dimension, T)
        trajs = self.add_noisy_diffusion(trajs, diffusion_coefficients)
        
        dataset[:, 2:] = trajs.reshape(dataset.shape[0], T*dimension)
        
        return dataset
    
    @staticmethod
    def add_noisy_localization(trajs, noise_func = False, sigma = 1, mu = 0):
        
        if isinstance(noise_func, np.ndarray):
            noise_matrix = noise_func 
        elif not noise_func:
            noise_matrix = sigma*np.random.randn(trajs.shape[0], trajs.shape[1])+mu
        elif hasattr(noise_func, '__call__'):
            noise_matrix = noise_func(trajs.shape[0], trajs.shape[1])             
        else:
            raise ValueError('noise_func has to be either False for Gaussian noise, a Python function or numpy array.')
        
        trajs += noise_matrix 
        
        return trajs
    
    @staticmethod
    def add_noisy_diffusion(trajs, diffusion_coefficients = False):
        
        # First normalize the trajectories
        trajs = normalize(trajs)
        # If no new diffusion coefficients given, create new ones randonmly
        if not diffusion_coefficients:
            diffusion_coefficients = np.random.randn(trajs.shape[0])
        # Apply new diffusion coefficients
        trajs = (trajs.transpose()*diffusion_coefficients).transpose()
        
        return trajs

    @staticmethod
    def create_segmented_dataset(dataset1, dataset2, dimension = 1, 
                                 final_length = 200, random_shuffle = False):
        ''' Creates a dataset with trajectories which change feature after a time
        't_change'. 
        Arguments:
            :dataset1 (numpy.array):
                - array of size Nx(t+2), where the first columns values correspond
                to the labels of the model and anomalous exponent. The rest 
                correspond to the trajectories of length t.
            :dataset2 (numpy.array):
                - same as dataset1
            :dimension (int):
                - Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
            :final_length (int):
                - length of the output trajectories.
            :random_shuffle (bool):
                - If True, shuffles the first axis of dataset1 and dataset2.
        Return:
            :seg_dataset (numpy.array):
                - array of size Nx(t+5) whose columns represent:
                    o Column 0: changing time
                    o Column 1,2: labels first part of the trajectory (model, exponent)
                    o Column 3,4: labels second part of the trajectory (model, exponent)
                    o Column 5:(t+5): trajectories of lenght t.'''
                    
        '''Establish dimensions and corresponding models'''                    
        
        if dataset1.shape[0] != dataset2.shape[0]:
            raise ValueError(f'Input datasets must have the same number of trajectories. Current ones have size {dataset1.shape[0]} and {dataset2.shape[0]}.')
        if dataset1.shape[1]-2 < final_length or dataset2.shape[1]-2 < final_length:
            raise ValueError(f'The trajectories in the input datasets are too short. They must be at least {final_length} steps long.')
        
        if random_shuffle:
            np.random.shuffle(dataset1)
            np.random.shuffle(dataset2)
        
        n_trajs = dataset1.shape[0]
        trajs_1 = np.copy(dataset1[:, 2:].reshape(n_trajs, dimension, int((dataset1.shape[1]-2)/dimension)))
        trajs_2 = np.copy(dataset2[:, 2:].reshape(n_trajs, dimension, int((dataset2.shape[1]-2)/dimension)))

        trajs_1 = trajs_1[:, :, :final_length]
        trajs_2 = trajs_2[:, :, :final_length]

        t_change = np.random.randint(1, final_length, n_trajs)

        seg_dataset = np.zeros((n_trajs, dimension*final_length+5))
        for idx, (tC, traj1, traj2, label1, label2) in enumerate(zip(t_change, 
                                                                      trajs_1, trajs_2,
                                                                      dataset1[:, :2], dataset2[:, :2])):
            seg_dataset[idx, 0] = tC
            seg_dataset[idx, 1:5] = np.append(label1, label2)

            if dimension == 1:
                seg_dataset[idx, 5:tC+5] = traj1[:, :tC]
                seg_dataset[idx, tC+5:] = traj2[:, tC:final_length]-traj2[:, tC]+traj1[:, tC]

            elif dimension == 2 or dimension == 3:
                traj2 = (traj2.transpose()-traj2[:, tC]+traj1[:, tC]).transpose()

                traj1[:,tC:]  = 0
                traj2[:, :tC] = 0

                seg_dataset[idx, 5:] = (traj1 + traj2).reshape(dimension*final_length)            
            
        return seg_dataset
    
    @staticmethod
    def save_row(data, file):
        '''Auxiliary function to save append data in existing files using csv
        Arguments:
            :data (numpy.array):
                - row to be appended to the filed
            :file (str):
                - file where to append data.'''
        with open(file, 'a') as f:
            writer = csv.writer(f, delimiter=';', lineterminator='\n',)
            writer.writerow(data)

    @staticmethod
    def cut_trajectory(traj, t_cut, dim=1):
        "Takes a trajectory and cuts it to `t_cut` length."
        cut_traj = traj.reshape(dim, -1)[:, :t_cut]
        return cut_traj.reshape(-1)    
    
    def andi_dataset(self, N = 1000, max_T = 1000, min_T = 10,
                           tasks = [1, 2, 3],
                           dimensions = [1, 2, 3],
                           load_dataset = False, save_dataset = False, path_datasets = '',
                           load_trajectories = False, save_trajectories = False, path_trajectories = 'datasets/',
                           N_save = 1000, t_save = 1000):  
        ''' Creates a dataset similar to the one given by in the ANDI challenge. 
        Check the webpage of the challenge for more details. The default values
        are similar to the ones used to generate the available dataset.
        Arguments:  
            :N (int, numpy.array):
                - if int, number of trajectories per class (i.e. exponent and model) in the dataset.
                - if numpy.array, number of trajectories per classes: size (number of models)x(number of classes)    
            :max_T (int):
                - Maximum length of the trajectories in the dataset.
            :min_T (int):
                - Minimum length of the trajectories in the dataset.
            :tasks (int, array):
                - Task(s) of the ANDI for which datasets will be generated. Task 1 corresponds to the
                anomalous exponent estimation, Task 2 to the model prediction and Task 3 to the segmen-
                tation problem.
            :dimensions (int, array):
                - Task(s) for which trajectories will be generated. Three possible values: 1, 2 and 3.
            :load_dataset (bool):
                - if True, the module loads the trajectories from the files task1.txt, task2.txt and
                task3.txt and the labels from ref1.txt, ref2.txt and ref3.txt. If the trajectories do
                not exist but the file does, the module returns and empty dataset.
            :save_dataset (bool):
                - if True, the module saves the datasets in a .txt following the format discussed in the 
                webpage of the comptetion.
            :load_trajectories (bool):
                - if True, the module loads the trajectories of an .h5 file.  
            :save_trajectories (bool):
                - if True, the module saves a .h5 file for each model considered, with N_save trajectories 
                  and T = T_save..
            :N_save (int):
                - Number of trajectories to save for each exponents/model. Advise: save at the beggining
                  a big dataset (i.e. with default t_save N_save) which allows you to load any other combiantion
                  of T and N.
            :t_save (int):
                - Length of the trajectories to be saved. See comments on N_save.                
        Return:
            The function returns 6 variables, three variables for the trajectories and three 
            for the corresponding labels. Each variable is a list of three lists. Each of the
            three lists corresponds to a given dimension, in ascending order. If one of the
            tasks/dimensions was not calculated, the given list will be empty
            :X1 (list of three lists):
                - Trajectories corresponding to Task 1. 
            :Y1 (list of three lists):
                - Labels corresponding to Task 1
            :X2 (list of three lists):
                - Trajectories corresponding to Task 2. 
            :Y2 (list of three lists):
                - Labels corresponding to Task 2
            :X3 (list of three lists):
                - Trajectories corresponding to Task 3. 
            :Y3 (list of three lists):
                - Labels corresponding to Task 3      '''
                    
        print(f'Creating a dataset for task(s) {tasks} and dimension(s) {dimensions}.')
        
        # Checking inputs for errors
        if isinstance(dimensions, int) or isinstance(dimensions, float):
            dimensions = [dimensions]
        if isinstance(tasks, int) or isinstance(tasks, float):
            tasks = [tasks]
        
        # Define return datasets
        X1 = [[],[],[]]; X2 = [[],[],[]]; X3 = [[],[],[]]
        Y1 = [[],[],[]]; Y2 = [[],[],[]]; Y3 = [[],[],[]]
        
        if load_dataset or save_dataset:
            # Define name of result files, if needed
            task1 = path_datasets+'task1.txt'; ref1 = path_datasets+'ref1.txt'
            task2 = path_datasets+'task2.txt'; ref2 = path_datasets+'ref2.txt'
            task3 = path_datasets+'task3.txt'; ref3 = path_datasets+'ref3.txt'
        
        # Loading the datasets if chosen.
        if load_dataset:            
            for idx, (task, lab) in enumerate(zip([task1, task2, task3], [ref1, ref2, ref3])):
                if idx+1 in tasks:
                    
                    try:
                        t = csv.reader(open(task,'r'), delimiter=';', 
                                        lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
                        l = csv.reader(open(lab,'r'), delimiter=';', 
                                        lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
                    except:
                        raise FileNotFoundError(f'File for task {idx+1} not found.')
                    
                    for trajs, labels in zip(t, l):   
                        if task == task1:                            
                            X1[int(trajs[0])-1].append(trajs[1:])
                            Y1[int(trajs[0])-1].append(labels[1])
                        if task == task2:
                            X2[int(trajs[0])-1].append(trajs[1:])
                            Y2[int(trajs[0])-1].append(labels[1])
                        if task == task3:
                            X3[int(trajs[0])-1].append(trajs[1:])
                            Y3[int(trajs[0])-1].append(labels[1:]) 
                    # Checking that the dataset exists in the files
                    for dim in dimensions:
                        if task == task1 and X1[dim-1] == []:
                            raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task1.txt.')
                        if task == task2 and X2[dim-1] == []:
                            raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task2.txt.')
                        if task == task3 and X3[dim-1] == []:
                            raise FileNotFoundError('Dataset for dimension '+str(dim)+' not contained in file task3.txt.')
                        
            return X1, Y1, X2, Y2, X3, Y3        

            
        exponents = np.arange(0.05, 2.01, 0.05)
        n_exp = len(exponents)
        # Trajectories per model and exponent. Arbitrarely chosen to obtain balanced classes
        n_per_model = np.ceil(1.6*N/5)
        subdif, superdif = n_exp//2, n_exp//2+1
        n_per_class =  np.zeros((self.n_models, n_exp))
        # ctrw, attm
        n_per_class[:2, :subdif] = np.ceil(n_per_model/subdif)
        # fbm
        n_per_class[2, :] = np.ceil(n_per_model/(n_exp-1))
        n_per_class[2, exponents == 2] = 0 # FBM can't be ballistic
        # lw
        n_per_class[3, subdif:] = np.ceil((n_per_model/superdif)*0.8)
        # sbm
        n_per_class[4, :] = np.ceil(n_per_model/n_exp)
        
        # Define return datasets
        X1 = [[],[],[]]; X2 = [[],[],[]]; X3 = [[],[],[]]
        Y1 = [[],[],[]]; Y2 = [[],[],[]]; Y3 = [[],[],[]]  
        
        # Initialize the files
        if save_dataset:
            if 1 in tasks:
                csv.writer(open(task1,'w'), delimiter=';', lineterminator='\n',)
                csv.writer(open(ref1,'w'), delimiter=';', lineterminator='\n',)
            elif 2 in tasks:
                csv.writer(open(task2,'w'), delimiter=';', lineterminator='\n',)
                csv.writer(open(ref2,'w'), delimiter=';',lineterminator='\n',)
            elif 3 in tasks:
                csv.writer(open(task3,'w'), delimiter=';', lineterminator='\n',)
                csv.writer(open(ref3,'w'), delimiter=';',lineterminator='\n',)
        
        for dim in dimensions:             
            # Generate the dataset of the given dimension
            print(f'Generating dataset for dimension {dim}.')
            dataset = self.create_dataset(T = max_T, N = n_per_class, exponents = exponents, 
                                           dimension = dim, models = np.arange(self.n_models),
                                           load_trajectories = False, save_trajectories = False, N_save = 100,
                                           path = path_trajectories)            
            
            # Normalize trajectories
            n_traj = dataset.shape[0]
            norm_trajs = normalize(dataset[:, 2:].reshape(n_traj*dim, max_T))
            dataset[:, 2:] = norm_trajs.reshape(dataset[:, 2:].shape)
    
            # Add localization error, Gaussian noise with sigma = [0.1, 0.5, 1]
                
            loc_error_amplitude = np.random.choice(np.array([0.1, 0.5, 1]), size = n_traj*dim)
            loc_error = (np.random.randn(n_traj*dim, int(max_T)).transpose()*loc_error_amplitude).transpose()
                        
            dataset = self.create_noisy_localization_dataset(dataset, dimension = dim, T = max_T, noise_func = loc_error)
            
            # Add random diffusion coefficients
            
            trajs = dataset[:, 2:].reshape(n_traj*dim, max_T)
            displacements = trajs[:, 1:] - trajs[:, :-1]
            # Get new diffusion coefficients and displacements
            diffusion_coefficients = np.random.randn(trajs.shape[0])
            new_displacements = (displacements.transpose()*diffusion_coefficients).transpose()  
            # Generate new trajectories and add to dataset
            new_trajs = np.cumsum(new_displacements, axis = 1)
            new_trajs = np.concatenate((np.zeros((new_trajs.shape[0], 1)), new_trajs), axis = 1)
            dataset[:, 2:] = new_trajs.reshape(dataset[:, 2:].shape)
            
        
            # Task 1 - Anomalous exponent
            if 1 in tasks:         
                # Creating semi-balanced datasets
                n_exp_max = int(np.ceil(1.1*N/n_exp))
                for exponent in exponents:
                    dataset_exp = dataset[dataset[:, 1] == exponent].copy()
                    np.random.shuffle(dataset_exp)
                    dataset_exp = dataset_exp[:n_exp_max, :]
                    try:
                        dataset_1 = np.concatenate((dataset_1, dataset_exp), axis = 0) 
                    except:
                        dataset_1 = dataset_exp
                np.random.shuffle(dataset_1)
                dataset_1 = dataset_1[:N, :]               
                
                for traj in dataset_1[:N, :]:             
                    # Cutting trajectories
                    cut_T = np.random.randint(min_T, max_T) 
                    traj_cut = self.cut_trajectory(traj[2:], cut_T, dim=dim).tolist()                         
                    # Saving dataset
                    X1[dim-1].append(traj_cut)
                    Y1[dim-1].append(np.around(traj[1], 2))
                    if save_dataset:                        
                        self.save_row(np.append(dim, traj_cut), task1)
                        self.save_row(np.append(dim, np.around([traj[1]], 2)), ref1)
            
            # Task 2 - Diffusion model
            if 2 in tasks:   
                # Creating semi-balanced datasets
                # If number of traejectories N is too small, consider at least
                # one trajectory per model
                n_per_model = max(1, int(1.1*N/5))
                    
                for model in range(5):
                    dataset_mod = dataset[dataset[:, 0] == model].copy()
                    np.random.shuffle(dataset_mod)
                    dataset_mod = dataset_mod[:n_per_model, :]
                    try:
                        dataset_2 = np.concatenate((dataset_2, dataset_mod), axis = 0) 
                    except:
                        dataset_2 = dataset_mod
                np.random.shuffle(dataset_2)
                dataset_2 = dataset_2[:N, :] 
                
                for traj in dataset_2[:N, :]:             
                    # Cutting trajectories
                    cut_T = np.random.randint(min_T, max_T) 
                    traj_cut = self.cut_trajectory(traj[2:], cut_T, dim=dim).tolist() 
                    # Saving dataset   
                    X2[dim-1].append(traj_cut)
                    Y2[dim-1].append(np.around(traj[0], 2))    
                    if save_dataset:
                        self.save_row(np.append(dim, traj_cut), task2)
                        self.save_row(np.append(dim, traj[0]), ref2)   
           
                     
            # Task 3 - Segmentated trajectories
            if 3 in tasks:  
                # Create a copy of the dataset and use it to create the 
                # segmented dataset
                dataset_copy1 = dataset.copy()
                dataset_copy2 = dataset.copy()
                
                # Shuffling the hard way
                order_dataset1 = np.random.choice(np.arange(n_traj), n_traj, replace = False)        
                order_dataset2 = np.random.choice(np.arange(n_traj), n_traj, replace = False)        
                dataset_copy1 = dataset_copy1[order_dataset1] 
                dataset_copy2 = dataset_copy1[order_dataset2] 
                
                seg_dataset = self.create_segmented_dataset(dataset_copy1, dataset_copy2, dimension = dim)        
                seg_dataset = np.c_[np.ones(n_traj)*dim, seg_dataset]     
                
                # Checking that there are no segmented trajectories with same exponent and model 
                # in each segment. First we compute the difference between labels
                diff = np.abs(seg_dataset[:, 2]-seg_dataset[:, 4]) + np.abs(seg_dataset[:, 3]-seg_dataset[:, 5])
                # Then, if there are repeated labels, we eliminate those trajectories
                while len(np.argwhere(diff == 0)) > 0: 
                    seg_dataset = np.delete(seg_dataset, np.argwhere(diff == 0), axis = 0)
                    # If the size of the dataset is too small, we generate new segmented trajectories
                    # and add them to the dataset
                    if seg_dataset.shape[0] < N:
                        
                        # Shuffling the hard way
                        new_order_dataset1 = np.random.choice(np.arange(n_traj), n_traj, replace = False)        
                        new_order_dataset2 = np.random.choice(np.arange(n_traj), n_traj, replace = False)        
                        dataset_copy1 = dataset_copy1[new_order_dataset1] 
                        dataset_copy2 = dataset_copy1[new_order_dataset2] 
                        
                        order_dataset1 = np.concatenate((order_dataset1, new_order_dataset1))
                        order_dataset2 = np.concatenate((order_dataset2, new_order_dataset2))
                        
                        aux_seg_dataset = self.create_segmented_dataset(dataset_copy1, dataset_copy2, dimension = dim) 
                        aux_seg_dataset = np.c_[np.ones(aux_seg_dataset.shape[0])*dim, aux_seg_dataset] 
                        seg_dataset = np.concatenate((seg_dataset, aux_seg_dataset), axis = 0)

                        diff = np.abs(seg_dataset[:, 2]-seg_dataset[:, 4]) + np.abs(seg_dataset[:, 3]-seg_dataset[:, 5])
                    else:
                        break        
                    
                    
                X3[dim-1] = seg_dataset[:N, 6:]
                Y3[dim-1] = seg_dataset[:N, :6]
                
                if save_dataset:
                    for label, traj in zip(seg_dataset[:N, :6], seg_dataset[:N, 6:]):
                        self.save_row(np.append(dim, traj), task3)
                        self.save_row(np.around(label, 2), ref3) 
                        
                        
        return X1, Y1, X2, Y2, X3, Y3
