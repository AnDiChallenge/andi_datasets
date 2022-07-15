from andi_datasets.datasets_phenom import datasets_phenom
import numpy as np
np.random.seed(0)
import stochastic
stochastic.random.seed(0)

dph = datasets_phenom()
num_experiments = 9
dics = []
for i in range(num_experiments):
    if i > 4:
        i = np.random.randint(0, 5, 1)
    
    dic = dph._get_dic_andi2(i+1)    
    dics.append(dic)
    
df_list, lab_t, lab_e = dph.challenge_2022_dataset(save_data = True,
                                                   dics = dics,
                                                   num_fovs = 1, path ='data/')