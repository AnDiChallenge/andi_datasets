# AUTOGENERATED! DO NOT EDIT! File to edit: ../source_nbs/lib_nbs/datasets_theory.ipynb.

# %% auto 0
__all__ = ['datasets_theory']

# %% ../source_nbs/lib_nbs/datasets_theory.ipynb 3
import numpy as np
import os
import inspect
import h5py
from tqdm.auto import trange
import csv

# %% ../source_nbs/lib_nbs/datasets_theory.ipynb 4
from .utils_trajectories import normalize
from .models_theory import models_theory as models_theory

# %% ../source_nbs/lib_nbs/datasets_theory.ipynb 6
class datasets_theory():
     
    def __init__(self):        
        '''
        This class generates, saves and loads datasets of theoretical trajectories simulated 
        from various diffusion models (available at andi_datasets.models_theory). 
        '''
        self._dimension = 1
        self._get_models()
        
    def _get_models(self):        
        '''Loading subclass of models'''
        if self._dimension == 1:
            self._models = models_theory._oneD()
        elif self._dimension == 2:
            self._models = models_theory._twoD()
        elif self._dimension == 3:
            self._models = models_theory._threeD()
        else:
            raise ValueError('Our current understanding of the physical world is three dimensional and so are the diffusion models available in this class')
                
        available_models = inspect.getmembers(self._models, inspect.ismethod)      
        self.avail_models_name = [x[0] for x in available_models]
        self.avail_models_func = [x[1] for x in available_models]
    
    def create_dataset(self, T, N_models, exponents, models,
                       dimension = 1,
                       save_trajectories = False, load_trajectories = False, 
                       path = 'datasets/',
                       N_save = 1000, T_save = 1000):        
        ''' 
        Creates a dataset of trajectories via the theoretical models defined in `.models_theory`. Check our tutorials for use cases of this function.
        
        Parameters
        ----------
        T : int
            Length of the trajectories.   
        N_models : int, numpy.array
            - if int, number of trajectories per class (i.e. exponent and model) in the dataset.
            - if numpy.array, number of trajectories per classes: size (number of models)x(number of classes)    
        exponents : float, array
            Anomalous exponents to include in the dataset. Allows for two digit precision.
        models : bool, int, list
            Labels of the models to include in the dataset. 
            Correspodance between models and labels is given by self.label_correspodance, defined at init.
            If int/list, choose the given models. If False, choose all of them.
        dimensions : int
            Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
        save_trajectories : bool
            If True, the module saves a .h5 file for each model considered, with N_save trajectories 
            and T = T_save.
        load_trajectories : bool
            If True, the module loads the trajectories of an .h5 file.
        path : str
            Path to the folder where to save/load the trajectories dataset.
        N_save : int
            Number of trajectories to save for each exponents/model. 
            Advise: save at the beggining a big dataset (t_save ~ 1e3 and N_save ~ 1e4) 
            which then allows you to load any other combiantion of T and N_models.
        t_save : int
            Length of the trajectories to be saved. See comments on N_save.                
        
        Returns
        -------
        numpy.array
                - Dataset of trajectories of lenght Nx(T+2), with the following structure:
                    o First column: model label 
                    o Second column: value of the anomalous exponent
                    o 2:T columns: trajectories
        '''
                    
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
        self._dimension = dimension
        self._get_models()
                
        '''Managing models to load'''       
        # Load from a list of models
        if isinstance(models, list): 
            self._models_name = [self.avail_models_name[idx] for idx in models]     
            self._models_func = [self.avail_models_func[idx] for idx in models]
        # Load from a single model
        elif isinstance(models, int) and not isinstance(models, bool):
            self._models_name = [self.avail_models_name[models]]
            self._models_func = [self.avail_models_func[models]]
        # Load all available models
        else: 
            self._models_name =  self.avail_models_name
            self._models_func =  self.avail_models_func
            
        '''Managing number of trajectory per class:
            - Defines array num_class as a function of N'''                            
        if isinstance(N_models, int): 
            n_per_class = N_models*np.ones((len(self._models_name), len(exponents)))
            
        elif type(N_models).__module__ == np.__name__: 
            if len(self._models_name) != N_models.shape[0] or len(exponents) != N_models.shape[1]:
                raise ValueError('Mismatch between the dimensions of N and the number of different classes.'+
                                 f'N must be either an int (balanced classes) or an array of length {len(models)}x'
                                 f'{len(exponents)} (inbalaced classes).') 
            n_per_class = N_models
        else:
            raise TypeError('Type of variable N not recognized.')
                    
        '''Defining default values for saved datasets''' 
        N_save = np.ones_like(n_per_class)*N_save
        # If the number of class of a given class is bigger than N_save, we
        # change the value of N_save for that particular class.
        N_save = np.max([N_save, n_per_class], axis = 0)      
                
        ''' Loading/Saving/Creating datasets'''
        if load_trajectories:
            data_models = self._load_trajectories(T = T,
                                                 exponents = exponents,
                                                 models_name = self._models_name,
                                                 dimension = self._dimension,
                                                 n_per_class = n_per_class,
                                                 path = path,
                                                 N_save = N_save,
                                                 t_save = t_save)
        elif save_trajectories:
            self._save_trajectories(exponents = exponents,
                                   dimension = self._dimension,
                                   models_name = self._models_name,
                                   models_func = self._models_func,
                                   path = path, 
                                   n_per_class = n_per_class,
                                   N_save = N_save,
                                   t_save = t_save)
            
            data_models = self._load_trajectories(T = T,
                                                 exponents = exponents,
                                                 dimension = self._dimension,
                                                 models_name = self._models_name,                                                 
                                                 n_per_class = n_per_class,
                                                 path = path,
                                                 N_save = N_save,
                                                 t_save = t_save)
            
        else:           
            data_models = self._create_trajectories(T = T,                                                   
                                                   exponents = exponents, 
                                                   dimension = self._dimension,
                                                   models_name = self._models_name,
                                                   models_func = self._models_func,
                                                   n_per_class = n_per_class)       
            
        return data_models
    
    def _load_trajectories(self, T, exponents, dimension, 
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
        self._dimension = dimension
        self._get_models()
            
        
        if isinstance(models_name, int):
            models_name = [models_name]
               
        for idx_m, name in enumerate(models_name):        
            hf = h5py.File(path+name+'.h5', 'r+')
            
            for idx_e, exp  in enumerate(exponents):
                
                name_dataset = f'{exp:.2f}_T_{t_save}_N_'+ \
                                str(int(N_save[idx_m, idx_e]))+f'_dim_{self._dimension}'  
                
                n = int(n_per_class[idx_m, idx_e])
                if n == 0:
                    continue
                
                try:
                    data = (hf.get(name_dataset)[()][:n,:self._dimension*T]) 
                except:
                    raise TypeError('The dataset you want to load does not exist.')
                    
                
                data = self._label_trajectories(trajs = data, model_name = name, exponent = exp)                
                            
                if idx_e + idx_m == 0:
                    dataset = data
                else:
                    dataset = np.concatenate((dataset, data), axis = 0) 
        return dataset
     
    def _save_trajectories(self, exponents, models_name, models_func, path, n_per_class,
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
        self._dimension = dimension
        self._get_models()        
        
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
                    name_dataset = f'{exp:.2f}_T_{t_save}_N_{n}_dim_{self._dimension}' 
                    
                    if name_dataset not in hf:  
                        
                        data = np.zeros((n, self._dimension*t_save))                           
                        # TQDM variables
                        tq = trange(n)
                        tq.set_postfix(saving = True, model = name, exponent = exp)
                        for i in tq:
                            data[i, :] = func(t_save, exp)                           
                            
                        hf.create_dataset(name_dataset, data=data)
                        
                    else:
                        print(f'The dataset for {name} with exponent {round(exp,3)}'
                                +' already exists, no need of saving it again.')
            
        
    def _create_trajectories(self, T, exponents, dimension, models_name, models_func, n_per_class):  
        ''' 
        Create a dataset for the exponents and models considered. 
        
        Parameters
        ---------- 
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
        Returns
        -------
            :dataset (numpy.array):
                - Dataset of trajectories of lenght (number of models)x(T+2), with the following structure:
                    o First column: model label.
                    o Second column: value of the anomalous exponent.
                    o 2:T columns: trajectories.'''
            
        for idx_m, (name, func) in enumerate(zip(models_name, models_func)):
            for idx_e, exp in enumerate(exponents):
                
                
                n = int(n_per_class[idx_m, idx_e])
                data = np.zeros((n, self._dimension*T))  
                for i in range(n):
                    data[i, :] = func(T, exp)
                    
                data = self._label_trajectories(trajs = data, model_name = name, exponent = exp)   
                
                if idx_e + idx_m == 0:
                    dataset = data
                else:
                    dataset = np.concatenate((dataset, data), axis = 0)
                
        return dataset
                
            
    def _label_trajectories(self, trajs, model_name, exponent):
        ''' Labels given trajectories given the corresponding label for the model and exponent.
        For models, the label correspond to the position of the model in self.avail_models_name.
        For exponents, the label if the value of the exponent.
        
        Parameters
        ----------
            :trajs (numpy array):
                - trajectories to label
            :model_name (str):
                - name of the model from which the trajectories are coming from.
            :exponent (float):
                - Anomalous exponent of the trajectories. 
        Returns
        -------
        numpy.array
            Labelled trajectoreis, with the following structure:
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
        ''' 
        Create a dataset of noisy trajectories. 
        This function creates trajectories with _create_trajectories and then adds given noise to them.        
        All parameters are the same as _create_trajectories but noise_func.
        
        Parameters
        ----------
        dataset : bool, numpy array
            If False, creates a dataset with the given parameters.
            If numpy array, dataset to which the function applies the noise.
        noise_func : bool, function
            If False, the noise added to the trajectories will be Gaussian distributed, with 
            variance sigma and mean value mu.
            If function, uses the given function to generate noise to be added to the trajectory.
            The function must have as input two ints, N and M and the output must be a matrix of size NxM.
        
        Returns
        -------
        numpy.array
            Dataset of trajectories of lenght Nx(T+2), with the following structure:
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
        trajs = self._add_noisy_localization(trajs, noise_func, sigma, mu)
        
        dataset[:, 2:] = trajs.reshape(dataset.shape[0], T*dimension)
        
        return dataset    
    
    def create_noisy_diffusion_dataset(self, 
                                       dataset = False,
                                       T = False, N = False, exponents = False, models = False, dimension = 1,
                                       diffusion_coefficients = False,
                                       save_trajectories = False, load_trajectories = False, 
                                       path = 'datasets/',
                                       N_save = 1000, t_save = 1000): 
        ''' 
        Create a dataset of noisy trajectories. 
        This function creates trajectories with `_create_trajectories` and then adds given noise to them.        
        All arguments are the same as `_create_trajectories` but dataset and diffusion_coefficients.
        
        Parameters
        ----------       
        dataset : bool, numpy array
                - If False, creates a dataset with the given parameters. 
                - If numpy array, dataset to which the function applies the noise.
        diffusion_coefficient : bool, function
                - If False, the diffusion noise added to the trajectories will 
                  be Gaussian distributed, with variance sigma and mean value mu.
                - If numpy array, multiply the displacements by them.
                
        Returns
        -------
        data_models : numpy.array
                Dataset of trajectories of lenght Nx(T+2), with the following structure:
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
        trajs = self._add_noisy_diffusion(trajs, diffusion_coefficients)
        
        dataset[:, 2:] = trajs.reshape(dataset.shape[0], T*dimension)
        
        return dataset
    
    @staticmethod
    def _add_noisy_localization(trajs, noise_func = False, sigma = 1, mu = 0):
        
        if isinstance(noise_func, np.ndarray):
            noise_matrix = noise_func 
        elif not noise_func:
            noise_matrix = sigma*np.random.randn(*trajs.shape)+mu
        elif hasattr(noise_func, '__call__'):
            noise_matrix = noise_func(trajs)             
        else:
            raise ValueError('noise_func has to be either False for Gaussian noise, a Python function or numpy array.')
        
        trajs += noise_matrix 
        
        return trajs
    
    @staticmethod
    def _add_noisy_diffusion(trajs, diffusion_coefficients = False, sigma = 1, mu = 0):
        
        # First normalize the trajectories
        trajs = normalize(trajs)
        # Check if diffusion coefficient are an array
        if isinstance(diffusion_coefficients, np.ndarray):
            pass
        # If no new diffusion coefficients given, create new ones randonmly
        elif not diffusion_coefficients:
            diffusion_coefficients = sigma*np.random.randn(trajs.shape[0])+mu
        # Apply new diffusion coefficients
        trajs = (trajs.transpose()*diffusion_coefficients).transpose()
        
        return trajs

    @staticmethod
    def create_segmented_dataset(dataset1, dataset2, dimension = 1, 
                                 final_length = 200, random_shuffle = False):
        ''' 
        Creates a dataset with trajectories which change diffusive feature (either model or anomalous exponent) after a time 't_change'. 
        
        Parameters
        ----------
        dataset1 : numpy.array
            Array of size Nx(t+2), where the first columns values correspond
            to the labels of the model and anomalous exponent. The rest 
            correspond to the trajectories of length t.
        dataset2 : numpy.array
            Same as dataset1
        dimension : int
            Dimensions of the generated trajectories. Three possible values: 1, 2 and 3.
        final_length : int
            Length of the output trajectories.
        random_shuffle : bool
            If True, shuffles the first axis of dataset1 and dataset2.
            
        Returns
        -------
        numpy.array
                Array of size Nx(t+5) whose columns represent:
                    o Column 0: changing time
                    o Column 1,2: labels first part of the trajectory (model, exponent)
                    o Column 3,4: labels second part of the trajectory (model, exponent)
                    o Column 5:(t+5): trajectories of lenght t.
        '''
                    
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
    def _save_row(data:np.array, # Row to be appended to the filed
                  file:str # File where to append data
                 ):
        ''' Auxiliary function to save append data in existing files using csv. '''
        
        with open(file, 'a') as f:
            writer = csv.writer(f, delimiter=';', lineterminator='\n',)
            writer.writerow(data)

    @staticmethod
    def _cut_trajectory(traj, t_cut, dim=1):
        ''' Takes a trajectory and cuts it to `t_cut` length. '''
        cut_traj = traj.reshape(dim, -1)[:, :t_cut]
        return cut_traj.reshape(-1)    
