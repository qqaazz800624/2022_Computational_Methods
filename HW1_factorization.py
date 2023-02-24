#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
#Use QR factorization to find all eigenvectors with REAL eigenvalues of X

def QR_factorization(a):
  dim = a.shape
  Q = np.zeros(dim)
  R = np.zeros((dim[1], dim[1]))

  R[0,0] = np.sqrt(sum(x**2 for x in a[:, 0].reshape(-1,1)))
  Q[:, 0] = a[:, 0]/R[0,0]

  for k in range(1, dim[1]):
    R[0:k, k] = [sum(Q[:, i]*a[:,k]) for i in range(k)]
    q = a[:, k] -sum(R[i,k]*Q[:, i] for i in range(k))
    R[k,k] = np.sqrt(sum([x**2 for x in q.reshape(-1,1)]))
    Q[:, k] = q/R[k,k]
  
  return Q, R


def QR_iteration(a, precision=0.000001, iter_ceil=3000):
    dim = a.shape
    real_value = []
    for i in range(dim[0]):
        times = 1
        a_0 = a.copy()
        Q_eigen = np.identity(dim[0])  # identity matrix
        
        while True:
            Q, R = QR_factorization(a_0)
            Q_eigen = np.dot(Q_eigen, Q) # converges to eigenvector
            ak = np.dot(R,Q) # converges to eigenvalue

            if np.abs(ak[i][i] - a_0[i][i]) < precision:
                print("Find an eigenvalue. Iteration times:", times)
                real_value.append(ak[i][i])
                break
            elif times > iter_ceil:
                print("The eigenvalue cannot converge.")
                break      
            else:
                times = times + 1
                a_0 = ak
                
    return Q_eigen, real_value


#%% 4.d Find all principal components with their eigenvalues using SVD


def SVD_factorization(a):
  m,n = a.shape
  S = np.zeros([m,n])
  U = np.zeros([m,m])
  data = np.dot(a.T, a)
  eigenvals, eigenvecs = np.linalg.eig(data)
  eig_index = eigenvals.argsort()[::-1]
  eigenval_desc = eigenvals[eig_index]
  V = eigenvecs[:, eig_index]

  #把 S 矩陣算出來
  for i in range(m):
    for j in range(n):
        if i == j:
            S[i,j] = np.sqrt(eigenval_desc[i])
  
  #把 U 矩陣算出來
  for i in range(m):
    if np.diagonal(S)[i] != 0:
        u = (1/np.diagonal(S)[i])*np.dot(a, V[:,i].reshape(n,1))
        u = u.reshape(m,)
        U[:,i] = u
    else:
        U[:,i] = np.zeros(m)
  
  return U, S, V


#%% 5.a 
import scipy.stats as stats
from scipy.stats import kurtosis
import pandas as pd
import numpy as np

data = pd.read_csv('/Users/wujhejia/Documents/Python/CAmaxTemp.txt', sep = '   ').reset_index(drop=False)
data['Station'] = data['index'].map(lambda x:x.split('\t')[0])
data['Period'] = data['index'].map(lambda x:x.split('\t')[1])
data.drop('index', axis=1, inplace=True)
data.columns = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','MAX','Station','Period']
data = data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
#data = data.to_numpy()

data_prime = data[['FEB','JUN','OCT']]
results = {}
months = {0:'FEB',1:'JUN',2:'OCT'}
for i in range(3):
  results[i] = kurtosis(data_prime[months[i]])
#%%
results


#%% 5.b

#Preprocessing Steps - Step1:Centering
def centering(data):
  output = data - data.mean()
  return output

D = centering(data_prime)

#Preprocessing Steps - Step2:Decorrelation
def decorrelation(D):
  D = D.to_numpy()
  D_cov = np.cov(D.T)
  eigs, V = np.linalg.eigh(D_cov)
  lam = np.diag(eigs)
  lam_inv = np.sqrt(np.linalg.inv(lam))
  U = np.dot(V, D.T).T
  return lam_inv, U

lam_inv, U = decorrelation(D)

#Preprocessing Steps - Step3:Whitening
# print(lam_inv.shape)
# print(U.shape)
Z = np.dot(lam_inv, U.T)
Z.T

#%% 5.c Graphical Illustration

# Scatter Plots - Original Data
#data_prime.plot.scatter(x='JUN',y='OCT',s=100)

# Scatter Plots - PCA space Projection
# plt.scatter(U[:,1],U[:,2])
# plt.show()

#Scatter Plots - After Whitening
plt.scatter(Z.T[:,1],Z.T[:,2])
plt.show()



#%% 5.d Fast ICA: Kurtosis Maximization

def fast_ICA1(a, precision=0.1**15, iter_ceil=100000):
    m,n = a.shape
    w0 = np.ones(n)
    #w0 = np.random.random(n)
    W = np.zeros(m)
    Z = np.zeros([m,n])
    times = 1
    while True:
        for i in range(m):
            W[i] = (w0.dot(a[i,]))**3
            Z[i,] = a[i,]*W[i]
        w1 = np.mean(Z, axis = 0) - 3*w0
        w1 = w1/np.linalg.norm(w1)
        if np.sum(np.abs(w1 - w0)) < precision:
            print('w converges. Iteration times:', times)
            break
        elif times > iter_ceil:
            print('w cannot converge.')
            break
        else:
            times = times + 1
            w0 = w1
    return w1

def fast_ICA2(a, u1, precision=0.1**15, iter_ceil=100000):
    m,n = a.shape
    w0 = np.ones(n)
    #w0 = np.random.random(n)
    W = np.zeros(m)
    Z = np.zeros([m,n])
    u1 = u1
    times = 1
    while True:
        for i in range(m):
            W[i] = (w0.dot(a[i,]))**3
            Z[i,] = a[i,]*W[i]
        w1 = np.mean(Z, axis = 0) - 3*w0
        w1 = w1/np.linalg.norm(w1)
        w1 = w1 - np.dot(w1,u1)*u1
        w1 = w1/np.linalg.norm(w1)
        if np.sum(np.abs(w1 - w0)) < precision:
        #if np.linalg.norm(w1 - w0) < precision:
            print('w converges. Iteration times:', times)
            break
        elif times > iter_ceil:
            print('w cannot converge.')
            break
        else:
            times = times + 1
            w0 = w1
    return w1

def fast_ICA3(a, u1, u2, precision=0.1**15, iter_ceil=100000):
    m,n = a.shape
    w0 = np.ones(n)
    #w0 = np.random.random(n)
    W = np.zeros(m)
    Z = np.zeros([m,n])
    u1 = u1
    u2 = u2
    times = 1
    while True:
        for i in range(m):
            W[i] = (w0.dot(a[i,]))**3
            Z[i,] = a[i,]*W[i]
        w1 = np.mean(Z, axis = 0) - 3*w0
        w1 = w1/np.linalg.norm(w1)
        w1 = w1 - np.dot(w1,u1)*u1 - np.dot(w1,u2)*u2
        w1 = w1/np.linalg.norm(w1)
        if np.sum(np.abs(w1 - w0)) < precision or np.sum(np.abs(w1)-np.abs(w0)) < precision:
        #if np.linalg.norm(w1 - w0) < precision:
            print('w converges. Iteration times:', times)
            break
        elif times > iter_ceil:
            print('w cannot converge.')
            break
        else:
            times = times + 1
            w0 = w1
    return w1

#%%

u1 = fast_ICA1(Z.T)
u2 = fast_ICA2(Z.T, u1)
u3 = fast_ICA3(Z.T, u1, u2)
W = np.array([u1,u2,u3])
print(W)


#%%

W = np.array([u1,u2,u3])
#np.round(W,4)
np.dot(W[1,],W[2,])


#%%

np.cross(u1,u2)



#%%