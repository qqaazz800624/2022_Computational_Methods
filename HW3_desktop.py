#%%
import numpy as np
from scipy.stats import randint, gamma, norm
import matplotlib.pyplot as plt
import math

#%% 1.c Poisson

def poisson(_lambda = 10): 
  x = 0
  u = np.random.uniform(0,1)
  exp_lambda = np.exp(-_lambda)
  poisson_cdf = exp_lambda 
  power = 1

  while u > poisson_cdf:
    x += 1
    factorial = math.factorial(x)
    power = _lambda * power
    poisson_cdf += (power / factorial) * exp_lambda

  return x, u

#%% 1.f Binomial

def bernoulli(p):
    if np.random.uniform(0,1) < p:
        return 1
    else:
        return 0

def binomial(n,p):
    var = 0
    for i in range(n):
        var = var + bernoulli(p)
    return var

#%% 1.g Negative Binomial

def NB(k,p):
    x = 0
    success = 0
    u = np.random.uniform(0,1)
    while success < k:
        if u < p:
            x = x + 1
            success= success+1
            u = np.random.uniform(0,1)
        else:
            x = x + 1
            u = np.random.uniform(0,1)
    if success == k:
        return x



#%% 3.a

def pi_estimate(n, h):
    x = [0]
    y = [0]
    t = 0
    while t <n - 1:
        xt = np.random.uniform(-h, h)
        yt = np.random.uniform(-h, h)
        try_x = xt + x[t]
        try_y = yt + y[t]
        if -1 <= try_x <= 1 and -1 <= try_y <= 1:
            x.append(try_x)
            y.append(try_y)
            t += 1
            #print(len(x))
    return x, y

def cal_pi(x, y, n):
    in_circle = 0
    for i in range(len(x)):
        if x[i]**2 + y[i]**2 <= 1:
            in_circle += 1
    pi = 4 * in_circle / n
    return pi


h_list = [1, 0.5, 0.2, 2, 3]
n_list = [20000, 25000, 30000, 40000]
for h in h_list:
    for n in n_list:
        x, y = pi_estimate(n, h)
        pi = cal_pi(x, y, n)
        print("n = ", n, "h = ", h, "pi = ", pi)

#%% 3.c

def mh_pi_estimate(n, h):
    x = [0]
    y = [0]
    t = 0
    while t < n - 1:
        xt = np.random.uniform(x[t] - h, x[t] + h)
        yt = np.random.uniform(y[t] - h, y[t] + h)
        try_x = xt
        try_y = yt
        #print(try_x)
        #print(try_y)
        if -1 <= try_x <= 1:
            x.append(try_x)
            #y.append(try_y)
        else:
            x.append(x[t])
            #y.append(y[t])
        if -1 <= try_y <= 1:
            y.append(try_y)
        else:
            y.append(y[t])
        t += 1
        #print(x[t], y[t])
            #print(len(x))
    return x, y

h_list = [1, 0.5, 0.2, 2, 3]
n_list = [20000, 25000, 30000, 40000]
for h in h_list:
    for n in n_list:
        x, y = mh_pi_estimate(n, h)
        pi = cal_pi(x, y, n)
        print("n = ", n, "h = ", h, "pi = ", pi)

#%% 4.a

'''
Step 1: Generate initial random sample X (sample size: n = 112)
theta
X1: 1 ~ theta (sample size: theta)
X2: theta+1 ~ 112 (sample size: 112 - theta)
X = X1 + X2 (sample size: n = 112)
'''
theta = np.random.randint(low=1,high=111, size=1)
a1 = np.random.gamma(shape=10,scale=0.1,size=1)
a2 = np.random.gamma(shape=10,scale=0.1,size=1)
lambda1 = np.random.gamma(shape=3, scale= 1/a1, size=1)
lambda2 = np.random.gamma(shape=3, scale= 1/a2, size=1)
X1 = np.random.poisson(lam = lambda1, size=theta)
X2 = np.random.poisson(lam = lambda2, size=112-theta)
X = np.concatenate((X1,X2),axis=None)


#%% 4.a

def generator(lambda1, lambda2, a1, a2):
    theta = int(np.random.randint(low=1,high=111, size=1))
    s=2
    a1 = float(np.random.gamma(shape=a1/s, scale=s, size=1))
    a2 = float(np.random.gamma(shape=a2/s, scale=s, size=1))
    lambda1 = float(np.random.gamma(shape=lambda1*s, scale=1/s, size=1))
    lambda2 = float(np.random.gamma(shape=lambda2*s, scale=1/s, size=1))
    return theta, lambda1, lambda2, a1, a2

def target(theta, lambda1, lambda2, a1, a2, X):
    if not (0 < theta < 112 and lambda1 > 0 and lambda2 > 0 and a1 > 0 and a2 > 0):
        return -np.inf

    a = np.log(np.exp(-theta*lambda1))
    b = np.log(np.exp(-(112-theta)*lambda2))
    c = np.log(lambda1**(sum([X[i] for i in range(int(theta))]) + 2))
    d = np.log(lambda2**(sum([X[i] for i in range(int(theta), len(X))])+2))
    e = np.log(np.exp(-(10+lambda1)*a1))
    f = np.log(np.exp(-(10+lambda2)*a2))
    g = np.log(a1**12)
    h = np.log(a2**12)

    return float(a+b+c+d+e+f+g+h)

def proposal(x1, x2, x3, x4, lambda1, lambda2, a1, a2):
    s = 2
    C = gamma.pdf(x=x3,a=a1/s,scale=s)
    D = gamma.pdf(x=x4,a=a2/s,scale=s)
    A = gamma.pdf(x=x1,a=lambda1*s,scale=1/s)
    B = gamma.pdf(x=x2,a=lambda2*s,scale=1/s)
    return float(A*B*C*D)

#%%

N = 500000
Y = np.zeros([N,5])
V = np.zeros([N,5])
accept = 1
r=0
theta_v0, lambda1_v0, lambda2_v0, a1_v0, a2_v0 = generator(lambda1, lambda2, a1, a2)
V[0] = [theta_v0, lambda1_v0, lambda2_v0, a1_v0, a2_v0]
Y[0] = V[0]

for i in range(1,N):
    epsilon = 1e-300
    theta_v, lambda1_v, lambda2_v, a1_v, a2_v = generator(lambda1=Y[i-1,1],lambda2=Y[i-1,2],a1=Y[i-1,3],a2=Y[i-1,4])
    V[i] = [theta_v, lambda1_v, lambda2_v, a1_v, a2_v]
    ratio_up = np.exp(target(V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], X))*proposal(x1=Y[i-1,1], x2=Y[i-1,2], x3=Y[i-1,3], x4=Y[i-1,4], lambda1=V[i,1], lambda2=V[i,2],a1=V[i,3],a2=V[i,4])
    ratio_down = np.exp(target(Y[i-1,0], Y[i-1,1], Y[i-1,2], Y[i-1,3], Y[i-1,4], X))*proposal(x1=V[i,1], x2=V[i,2], x3=V[i,3], x4=V[i,4], lambda1=Y[i-1,1], lambda2=Y[i-1,2],a1=Y[i-1,3],a2=Y[i-1,4])
    log_ratio = np.log(ratio_up)-np.log(ratio_down)
    rho = min(log_ratio,1)
    U = float(np.random.uniform(low=0,high=1,size=1))
    if np.log(U) <= rho:
        Y[i] = V[i]
        accept = accept+1
    else:
        Y[i] = Y[i-1]

print('Posterior mean of theta:',Y[:,0].mean())
print('Posterior mean of lambda1:',Y[:,1].mean())
print('Posterior mean of lambda2:',Y[:,2].mean())


#%% Trace plots in 4.a

#plt.plot(range(N), Y[:,0], label='theta')
#plt.plot(range(N), Y[:,1], label='lambda1')
#plt.plot(range(N), Y[:,2], label='lambda2')

plt.title('Traceplot')
plt.ylabel("Estimate")
plt.xlabel("Steps")
plt.legend()
plt.show()

#%% Histograms in 4.a

#plt.hist(Y[:,0], bins=30, label='Theta')
#plt.hist(Y[:,1], bins=30, label='Lambda1')
plt.hist(Y[:,2], bins=30, label='Lambda2')
plt.title('Histogram')
plt.xlabel('Samples of Parameter')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#%% 4.b

logalpha = np.random.uniform(low=np.log(1/8), high=np.log(2),size=1)
alpha = np.exp(logalpha)
theta = np.random.randint(low=1,high=111, size=1)
a = np.random.gamma(shape=10,scale=0.1,size=1)
lambda1 = np.random.gamma(shape=3, scale= 1/a, size=1)
lambda2 = alpha*lambda1
X1 = np.random.poisson(lam = lambda1, size=theta)
X2 = np.random.poisson(lam = lambda2, size=112-theta)
X = np.concatenate((X1,X2),axis=None)


#%%


def generator2(lambda1, a):
    theta = int(np.random.randint(low=1,high=111, size=1))
    logalpha = np.random.uniform(low=np.log(1/8), high=np.log(2),size=1)
    alpha = float(np.exp(logalpha))
    s = 2
    a = float(np.random.gamma(shape=a/s,scale=s,size=1))
    lambda1 = float(np.random.gamma(shape=lambda1*s, scale= 1/s, size=1))
    return theta, lambda1, alpha, a


def target2(theta, lambda1, alpha, a, X):
    if not (0 < theta < 112 and lambda1 > 0 and lambda2 > 0 and 1/8 <= alpha <= 2 and a > 0):
        return -np.inf

    a1 = np.log(np.exp(-theta*lambda1))
    b1 = np.log(np.exp(-(112-theta)*alpha*lambda1))
    c1 = np.log(lambda1**(sum([X[i] for i in range(int(theta))]) + 2))
    d1 = np.log((alpha*lambda1)**(sum([X[i] for i in range(int(theta), len(X))])))
    e1 = np.log(np.exp(-(10+lambda1)*a))
    f1 = np.log(a**12)
    g1 = np.log((1/np.log(16))*(1/alpha))
    #g1 = np.log(np.exp(np.random.uniform(low=-np.log(8), high=np.log(2),size=1)))
    return float(a1+b1+c1+d1+e1+f1+g1)


def alpha_pdf(alpha):
    if not(1/8 <= alpha <= 2):
        return -np.inf
    out = (1/np.log(16))*(1/alpha)
    return out

def proposal2(lambda_x, a_x, lambda1, alpha, a):
    s=2
    A=gamma.pdf(x=lambda_x, a=lambda1*s, scale=1/s)
    B=gamma.pdf(x=a_x, a=a/s, scale=s)
    C = alpha_pdf(alpha)
    return float(A*B*C)

#%%

N = 200000
Y = np.zeros([N,4])
V = np.zeros([N,4])
accept = 1
theta_v0, lambda1_v0, alpha_v0, a_v0 = generator2(lambda1, a)
V[0] = [theta_v0, lambda1_v0, alpha_v0, a_v0]
Y[0] = V[0]


for i in range(1,N):
    theta_v, lambda1_v, alpha_v, a_v = generator2(lambda1=Y[i-1,1],a=Y[i-1,3])
    V[i] = [theta_v, lambda1_v, alpha_v, a_v]
    ratio_up = np.exp(target2(V[i,0], V[i,1], V[i,2], V[i,3], X))*proposal2(lambda_x=Y[i-1,1], a_x=Y[i-1,3],lambda1=V[i,1],alpha=V[i,2],a=V[i,3])
    ratio_down = np.exp(target2(Y[i-1,0], Y[i-1,1], Y[i-1,2], Y[i-1,3], X))*proposal2(lambda_x=V[i,1], a_x=V[i,3],lambda1=Y[i-1,1],alpha=Y[i-1,2],a=Y[i-1,3])
    log_ratio = np.log(ratio_up)-np.log(ratio_down)
    rho = min(log_ratio,1)
    U = float(np.random.uniform(low=0,high=1,size=1))
    if np.log(U) <= rho:
        Y[i] = V[i]
        accept = accept+1
    else:
        Y[i] = Y[i-1]


print('Posterior mean of theta:',Y[:,0].mean())
print('Posterior mean of lambda1:',Y[:,1].mean())
print('Posterior mean of lambda2:',(Y[:,1]*Y[:,2]).mean())
print('Posterior mean of a:',Y[:,3].mean())


#%% Traceplots in 4.b

#plt.plot(range(N), Y[:,0], label='theta')
#plt.plot(range(N), Y[:,1], label='lambda1')
#plt.plot(range(N), Y[:,1]*Y[:,2], label='lambda2')
# plt.plot(range(N), Y[:,3], label='a')


plt.ylabel("Estimate")
plt.xlabel("Steps")
plt.legend()
plt.show()

#%% Histograms in 4.b

#plt.hist(Y[:,0], bins=20, label='Theta')
#plt.hist(Y[:,1], bins=20, label='Lambda1')
plt.hist(Y[:,1]*Y[:,2], bins=20, label='Lambda2')
plt.title('Histogram')
plt.xlabel('Samples of Parameter')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#%% 4.c

theta = np.random.randint(low=1,high=111, size=1)
a1 = np.random.uniform(0,100,size=1)
a2 = np.random.uniform(0,100,size=1)
lambda1 = np.random.gamma(shape=3, scale= 1/a1, size=1)
lambda2 = np.random.gamma(shape=3, scale= 1/a2, size=1)
X1 = np.random.poisson(lam = lambda1, size=theta)
X2 = np.random.poisson(lam = lambda2, size=112-theta)
X = np.concatenate((X1,X2),axis=None)

#%%

def generator3(lambda1,lambda2):
    theta = int(np.random.randint(low=1,high=111, size=1))
    a1 = float(np.random.uniform(0,100,size=1))
    a2 = float(np.random.uniform(0,100,size=1))
    s=2
    lambda1 = float(np.random.gamma(shape=lambda1*a1, scale= 1/a1, size=1))
    lambda2 = float(np.random.gamma(shape=lambda1*a1, scale= 1/a2, size=1))
    return theta, lambda1, lambda2, a1, a2

def target3(theta, lambda1, lambda2, a1, a2, X):
    A = np.log(np.exp(-theta*lambda1))
    B = np.log(np.exp(-(112-theta)*lambda2))
    C = np.log(lambda1**(sum([X[i] for i in range(int(theta))]) + 2))
    D = np.log(lambda2**(sum([X[i] for i in range(int(theta), len(X))])+2))
    E = np.log((a1*a2)**3)
    F = np.log(np.exp(-(a1*lambda1+a2*lambda2)))
    return float(A+B+C+D+E+F)

def proposal3(lambda1_x, lambda2_x, lambda1, lambda2):
    s=2
    A=gamma.pdf(x=lambda1_x, a=lambda1*s, scale=1/s)
    B=gamma.pdf(x=lambda2_x, a=lambda2*s, scale=1/s)
    return float(A*B)

#%%

N = 500000
Y = np.zeros([N,5])
V = np.zeros([N,5])
accept = 1
theta_v0, lambda1_v0, lambda2_v0, a1_v0, a2_v0 = generator3(lambda1,lambda2)
V[0] = [theta_v0, lambda1_v0, lambda2_v0, a1_v0, a2_v0]
Y[0] = V[0]

for i in range(1,N):
    theta_v, lambda1_v, lambda2_v, a1_v, a2_v = generator3(lambda1=Y[i-1,1],lambda2=Y[i-1,2])
    V[i] = [theta_v, lambda1_v, lambda2_v, a1_v, a2_v]
    ratio_up = np.exp(target3(V[i,0], V[i,1], V[i,2], V[i,3], V[i,4], X))*proposal3(lambda1_x=Y[i-1,1],lambda2_x=Y[i-1,2],lambda1=V[i,1],lambda2=V[i,2])
    ratio_down = np.exp(target3(Y[i-1,0], Y[i-1,1], Y[i-1,2], Y[i-1,3], Y[i-1,4], X))*proposal3(lambda1_x=V[i,1],lambda2_x=V[i,2],lambda1=Y[i-1,1],lambda2=Y[i-1,2])
    log_ratio = np.log(ratio_up)-np.log(ratio_down)
    rho = min(log_ratio,1)
    U = float(np.random.uniform(low=0,high=1,size=1))
    if np.log(U) <= rho:
        Y[i] = V[i]
        accept = accept+1
    else:
        Y[i] = Y[i-1]


print('Posterior mean of theta:',Y[:,0].mean())
print('Posterior mean of lambda1:',Y[:,1].mean())
print('Posterior mean of lambda2:',Y[:,2].mean())


#%% Traceplots in 4.c

plt.plot(range(N), Y[:,0], label='theta')
#plt.plot(range(N), Y[:,1], label='lambda1')
#plt.plot(range(N), Y[:,2], label='lambda2')

plt.title('Traceplot')
plt.ylabel("Estimate")
plt.xlabel("Steps")
plt.legend()
plt.show()

#%% Histograms in 4.c

#plt.hist(Y[:,0], bins=20, label='Theta')
#plt.hist(Y[:,1], bins=20, label='Lambda1')
plt.hist(Y[:,2], bins=20, label='Lambda2')
plt.title('Histogram')
plt.xlabel('Samples of Parameter')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#%% 4.e Generating Data

theta = np.random.randint(low=1,high=111, size=1)
a1 = np.random.gamma(shape=10,scale=0.1,size=1)
a2 = np.random.gamma(shape=10,scale=0.1,size=1)
lambda1 = np.random.gamma(shape=3, scale= 1/a1, size=1)
lambda2 = np.random.gamma(shape=3, scale= 1/a2, size=1)
X1 = np.random.poisson(lam = lambda1, size=theta)
X2 = np.random.poisson(lam = lambda2, size=112-theta)
X = np.concatenate((X1,X2),axis=None)


#%%

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

#plt.plot(range(N), Y[:,0], label='theta')
#plt.plot(range(N), Y[:,1], label='lambda1')
plt.plot(range(N), Y[:,2], label='lambda2')

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





#%%





#%%




#%%