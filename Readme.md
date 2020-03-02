# The Anomalous Diffusion (AnDi) Challenge

This repository contains the necessary functions to generate datasets of trajectories for the [**Anomalous Diffusion (AnDi) Challenge**](https://competitions.codalab.org/competitions/23601). The class AnDi allows to generate a dataset of trajectories generated according to various diffusion models, either in one, two or three dimensions. 

**Libraries needed**

- [math](https://docs.python.org/3/library/math.html) and [numpy](https://numpy.org/) for mathematical and array operations.
- [fbm](https://pypi.org/project/fbm/) to simulate the Fractional Brownian motion model.
- [scipy](https://www.scipy.org/) for the [erfcinv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcinv.html) function.
- [h5py](https://www.h5py.org/) for file management.
- [tqdm](https://github.com/tqdm/tqdm) for progress bars.

## Functions organization

The `ANDI` class allows to generate, save and load trajectories generated with various diffusion models. Its main purpose is to generate datasets similar to the ones proposed in the ANDI challenge. Moreover, it also allows to generate datasets of trajectories of various kinds, depending on the needs of the user. 

Examples of use of the class and more detailed descriptions of all the functions can be found in the notebook [`examples_andi.ipynb`](https://github.com/AnDiChallenge/ANDI_datasets/blob/master/examples_andi.ipynb).


## Available models:

The diffusion models considered in ANDI are all stored in the file `diffusion_models.py`. They are organized in the class with same name and separated in three subclasses, depending of the dimensions of the mode: `oneD`, `twoD`, `threeD`.
- One dimension
    - Continuous time random walk
    - Fractional Brownian motion
    - Lévy walk
    - Annealead transit time
    - Scaled Brownian motion
- Two dimensions
    - Continuous time random walk
    - Fractional Brownian motion
    - Lévy walk
    - Annealead transit time
    - Scaled Brownian motion
- Three dimensions
    - Continuous time random walk
    - Fractional Brownian motion
    - Lévy walk
    - Annealead transit time
    - Scaled Brownian motion

### Properties of diffusion models

Each of the models available in the AnDi class are created by a function which has the properties introduced below (please follow this same structure, also if you want to contribute additional diffusion models).

- **Keyboard inputs** (i.e. compulsory inputs, the functions may have as many optional inputs as you want):
    - `T`: the length of the trajectory.
    - `alpha`: the anomalous exponent. 
    
Important: in this class we are defining the anomalous exponent as the one calculated throught a ensemble averaged mean squared displacement over many trajectories. 

- **Output:**
    - `trajectory`: numpy.array of size `1x(d.T)`, where `d` is the number of dimensions.
    
Other outputs via optional inputs can be added at your discretion.


- **Other constrains:**
    - The trajectories must represent the position of a particle at regular times. If your model is constructed with irregular sampling, you can use the function `regularize(x,t,T)` placed in `utils.py` , where `x` is an array of the positions at sampling times contained in the array `t`. The output of such function is a numpy array of length `T`.
    - All the trajectories must start at 0.
    - Add your model to the subclass corresponding to the dimensions of your model. Add it *after* the last model of the subclass.
    - If you contribute with a model, insert its name in the list of models presented above and in the initial comment of the file `diffusion_models.py`.
 
