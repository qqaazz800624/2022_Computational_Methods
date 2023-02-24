#%%
import numpy as np
import random
import os


#%% 2. Basic Statistics in Roulette
import numpy as np
import random

def spins():
    slots = {'00': 'green', '0': 'green', '1': 'red', '2': 'black',
             '3': 'red', '4': 'black', '5': 'red', '6': 'black', 
             '7': 'red',
             '8': 'black', '9': 'red', '10': 'black', '11': 'red',
             '12': 'black', '13': 'red', '14': 'black', '15': 'red',
             '16': 'black', '17': 'red', '18': 'black', '19': 'red',
             '20': 'black', '21': 'red', '22': 'black', '23': 'red',
             '24': 'black', '25': 'red', '26': 'black', '27': 'red',
             '28': 'black', '29': 'red', '30': 'black', '31': 'red',
             '32': 'black', '33': 'red', '34': 'black', '35': 'red',
             '36': 'black'}
    result = random.choice(list(slots.keys()))
    return result



result = {}
for i in range(13):
    result[i] = spins()

result 

#%%

result = {}
for i in range(2):
    result[i] = spins()

result 

#%% 4. PCA and SVD


print (os.path.abspath('.'))

#%%
#讀取 txt 檔

path = '\Users\wujhejia\Desktop\CAmaxTemp.txt'
with open(file=path, mode='r') as f:
    data = f.read()

data

#%%

import numpy as np

def power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    
    eig_val = np.dot(np.dot(b_k.T,A),b_k)/np.dot(b_k.T, b_k)
    eig_vec = b_k
    return eig_val, eig_vec

power_iteration(np.array([[0.5, 0.5], [0.2, 0.8]]), 1000)
#%%



#%%
A = np.array([[0.5, 0.5], [0.2, 0.8]])
np.linalg.eig(A)

#%%




#%%
