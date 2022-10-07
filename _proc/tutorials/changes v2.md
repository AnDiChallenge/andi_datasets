# Changes in ANDI 1.0

- class `andi` $\rightarrow$ `andi_theory`

- function `andi.andi_dataset` $\rightarrow$ `andi_theory.challenge_2020_dataset`


- variable name change in `andi_theory.create_dataset`: `N` $\rightarrow$ `N_model`. In this way, it is more clear that `N_model` refers to number of trajectories per model and we use `N` for total number of trajectories in other functions

- corrected how noise is applied in Task3. Now the noise is added after the segmentation.

- Added an extra variable in `andi_theory.challenge_2020_dataset`, `return_noise` which, if `True`, makes the function output the noise amplitudes added to each trajectory

- Change how noise is applied in `andi_theory.challenge_2020_dataset`, such that all components of trajectories with dimension 2 and 3 have the same noise amplitude.

## To do:
- [ ] simplifiy num_per_class such that N_models can only be int (see datasets_phenom)