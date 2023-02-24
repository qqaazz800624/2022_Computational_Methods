#%%
import numpy as np
from scipy.stats import randint, gamma
import matplotlib.pyplot as plt

#%% 4.e

theta = np.random.randint(low=1,high=111, size=1)
a1 = np.random.gamma(shape=10,scale=0.1,size=1)
a2 = np.random.gamma(shape=10,scale=0.1,size=1)
lambda1 = np.random.gamma(shape=3, scale= 1/a1, size=1)
lambda2 = np.random.gamma(shape=3, scale= 1/a2, size=1)
X1 = np.random.poisson(lam = lambda1, size=theta)
X2 = np.random.poisson(lam = lambda2, size=112-theta)
X = np.concatenate((X1,X2),axis=None)

#%% Posterior Theta Generator 

class Theta(object):
    def __init__(self, lambda1, lambda2, X):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.X = X

    def theta_up(self, theta):
        A = np.exp(theta*(self.lambda2 - self.lambda1))
        B = (self.lambda1/self.lambda2)**(sum([self.X[i] for i in range(int(theta))]))
        return float(A*B)

    def theta_pmf(self, theta):
        A = self.theta_up(theta)
        B = sum([self.theta_up(theta = i) for i in range(1,112)])
        out = A/B
        return float(out) 

    def theta_cdf(self, theta):
        out = sum([self.theta_pmf(theta = i) for i in range(1,theta+1)])
        return out

    def generator(self):
        U = np.random.uniform(low=0, high=1, size=1)
        t = 1
        while self.theta_cdf(t) < U:
            t = t + 1
        return t


#%% Gibbs Sampler

theta_y0 = int(np.random.randint(low=1,high=111, size=1))
a1_y0 = float(np.random.gamma(shape=10,scale=0.1,size=1))
a2_y0 = float(np.random.gamma(shape=10,scale=0.1,size=1))
lambda1_y0 = float(np.random.gamma(shape=3, scale= 1/a1, size=1))
lambda2_y0 = float(np.random.gamma(shape=3, scale= 1/a2, size=1))
ThetaGenerator = Theta(lambda1, lambda2, X)
ThetaGenerator.generator()

N = 200
Y = np.zeros([N, 5])
Y[0] = [theta_y0, lambda1_y0, lambda2_y0, a1_y0, a2_y0]

for t in range(1, N):
    ThetaGenerator = Theta(lambda1 = Y[t-1,1], lambda2 = Y[t-1,2], X = X)
    Y[t,0] = ThetaGenerator.generator()
    Y[t,1] = np.random.gamma(shape=(sum([X[i] for i in range(int(Y[t,0]))])+3), scale = 1/(Y[t,0]+Y[t-1,3]), size=1)
    Y[t,2] = np.random.gamma(shape=(sum([X[i] for i in range(int(Y[t,0]), len(X))])+3), scale=1/(112-Y[t,0]+Y[t-1,4]), size=1)
    Y[t,3] = np.random.gamma(shape = 13, scale = 1/(10+Y[t,1]), size=1)
    Y[t,4] = np.random.gamma(shape = 13, scale = 1/(10+Y[t,2]), size=1)


#%% Traceplots in 4.e

plt.plot(range(N), Y[:,0], label='theta')
#plt.plot(range(N), Y[:,1], label='lambda1')
#plt.plot(range(N), Y[:,2], label='lambda2')

plt.ylabel("Estimate")
plt.xlabel("Steps")
plt.legend()
plt.show()

#%% Histograms in 4.f

#plt.hist(Y[:,0], bins=30, label='Theta')
#plt.hist(Y[:,1], bins=30, label='Lambda1')
plt.hist(Y[:,2], bins=30, label='Lambda2')
plt.title('Histogram')
plt.xlabel('Samples of Parameter')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#%% Summary Statistics in 4.f

print('Parameter: Theta','\nMean:',Y[:,0].mean(),'Median:', np.median(Y[:,0]), 'Min:', Y[:,0].min(), 'Max:', Y[:,0].max())
print('Parameter: Lambda1','\nMean:',Y[:,1].mean(),'Median:', np.median(Y[:,1]), 'Min:', Y[:,1].min(), 'Max:', Y[:,1].max())
print('Parameter: Lambda2','\nMean:',Y[:,2].mean(),'Median:', np.median(Y[:,2]), 'Min:', Y[:,2].min(), 'Max:', Y[:,2].max())


#%%