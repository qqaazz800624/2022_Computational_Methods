#%%

import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
from numpy.random import choice as np_choice
from itertools import combinations
from sys import maxsize
import math
import time

#%% 2
np.set_printoptions(linewidth=500, suppress=True)
path = '/Users/wujhejia/Documents/Python/711_distance_Brian.xlsx'
distance_df = pd.read_excel(path, header=0, skiprows = 0, usecols = "B : AF")
distance_array = distance_df.to_numpy()
city_df = pd.read_excel(path, index_col=0)
city_data = list(city_df.columns)


#%% 3.a

def find(target=5e8):
  x = 0
  while math.factorial(x) <= target:
    x = x + 1
    if math.factorial(x) > target:
      break
  return x-1, math.factorial(x-1)

find()


#%% 3.b

def random_path(graph):
  N = 30
  path = []
  cities_No = list(range(len(graph)))
  for i in range(1,N+1):
    randval = random.randint(1, len(cities_No)-1)
    randomCity = cities_No[randval]
    path.append(randomCity)
    cities_No.remove(randomCity)
  return path


path = random_path(graph = distance_array)
np.array(path)

#%% 3.c 

def path_distance(graph, path):
  N = len(path)
  distance = graph[0, path[0]]
  for i in range(N-1):
    distance = distance + graph[path[i], path[i+1]]
  distance = distance + graph[path[N-1], 0]
  return distance

dist_rec = np.zeros(1000)
for i in range(1000):
  path_temp = random_path(graph = distance_array)
  dist_tmp = path_distance(graph = distance_array, path = path_temp)
  dist_rec[i] = dist_tmp

dist_rec.mean()

#%% 3.d

class RejectionSampling(object):
  def __init__(self, threshold, num_samples):
    self.threshold = threshold
    self.num_samples = num_samples


  def random_path(self, graph):
    N = 30
    path = []
    cities_No = list(range(len(graph)))

    for i in range(1,N+1):
      randval = random.randint(1, len(cities_No)-1)
      randomCity = cities_No[randval]
      path.append(randomCity)
      cities_No.remove(randomCity)

    return path


  def path_distance(self, graph, path):
    N = len(path)
    distance = graph[0, path[0]]

    for i in range(N-1):
      distance = distance + graph[path[i], path[i+1]]
  
    distance = distance + graph[path[N-1], 0]
    return distance


  def good_path_gen(self):
    init_path = []

    while len(init_path) < self.num_samples:
      path_tmp = self.random_path(distance_array)
      dist_tmp = self.path_distance(distance_array, path_tmp)
      is_the_same = [(path_tmp == s).all() for s in init_path]
      if True not in is_the_same:
        if dist_tmp <= self.threshold:
          init_path.append(np.array(path_tmp))
        else:
          prob = random.random()
          if prob > 0.7:
            init_path.append(np.array(path_tmp))

    return init_path #np.array(init_path) 

RejectionSampler = RejectionSampling(threshold=155033, num_samples=1000)
GoodPaths = RejectionSampler.good_path_gen()
np.array(GoodPaths)


#%% 4. Simulated Annealing
'''
Pseudo-code

s = s0; e = E(s)                           #設定目前狀態為s0 其能量E(s0)
t = 0                                      #評估次數t
while t < tmax and e > emin                #若還有時間 評估次數t還不到tmax 且結果還不夠好 能量e不夠低 則：
    sn = neighbour(s)                         #隨機選取一鄰近狀態sn
    en = E(sn)                                #sn的能量為E (sn)
    if random() < P(e, en, temp(t/tmax))       #決定是否移至鄰近狀態sn
        s = sn; e = en                          #移至鄰近狀態sn
    t = t + 1                                  #評估完成 次數k加一
return s                                   #回傳狀態s

'''


start_time = time.time()

RejectionSampler = RejectionSampling(threshold=155033, num_samples=1000)
GoodPaths = RejectionSampler.good_path_gen()

class SimAnn(object):
    def __init__(self, n_iterations):
        """
        args:
            n_iteration (int): Number of iterations
        """
        self.n_iterations = n_iterations

    
    def random_path(self, graph):
      N = 30
      path = []
      cities_No = list(range(len(graph)))
      for i in range(1,N+1):
        randval = random.randint(1, len(cities_No)-1)
        randomCity = cities_No[randval]
        path.append(randomCity)
        cities_No.remove(randomCity)
      return path


    def path_distance(self, graph, path):
      N = len(path)
      distance = graph[0, path[0]]

      for i in range(N-1):
        distance = distance + graph[path[i], path[i+1]]
  
      distance = distance + graph[path[N-1], 0]
      return distance

    def getNeighbours(self, solution):
        a = np.random.choice(a = range(len(solution)),size = 2, replace=False)
        pos1 = a[0]
        pos2 = a[1]
        neighbor = solution.copy()
        neighbor[pos1] = solution[pos2]
        neighbor[pos2] = solution[pos1]
        return neighbor
    
    # def getNeighbours(self, solution):
    #     neighbours = []
    #     for i in range(len(solution)):
    #         for j in range(i + 1, len(solution)):
    #             neighbour = solution.copy()
    #             neighbour[i] = solution[j]
    #             neighbour[j] = solution[i]
    #             neighbours.append(neighbour)
    #     return neighbours


    def getTemp(self, t, temp):
        out = (1 - t/(1+self.n_iterations))*temp
        return out

    
    def run(self, graph, init_x, e=1e-10):
        t = 0
        T0 = 100

        pcur_x = np.array(init_x)
        pcur_y = self.path_distance(graph, path=pcur_x)
        pb_x, pb_y = pcur_x, pcur_y

        x_list = []
        y_list = []

        while t < self.n_iterations:
            # pcurx_neighbors = self.getNeighbours(pcur_x)
            # neighbor_idx = random.randint(0,len(pcurx_neighbors)-1)
            pnew_x = self.getNeighbours(pcur_x)
            pnew_y = self.path_distance(graph, path=pnew_x)
            dE = pnew_y - pcur_y

            if dE <= 0:
                pcur_x, pcur_y = pnew_x, pnew_y  
                if pcur_y < pb_y:
                    pb_x, pb_y = pcur_x, pcur_y
            else:
                T = self.getTemp(t, T0)
                T0 = T
                if np.random.random(1) < np.exp(-dE/(T+e)):
                    pcur_x, pcur_y = pnew_x, pnew_y 

                t = t + 1
                x_list.append(t)
                y_list.append(pb_y)

        return pb_x, pb_y, x_list, y_list


SimAnnealler = SimAnn(n_iterations = 1000)   

dict = {}

for i in range(1000):
  res_x, res_y, SA_x, SA_y = SimAnnealler.run(graph=distance_array, init_x=GoodPaths[i])
  dict[i] = {'Path': res_x, 'Distance': res_y}

dist_rec_array = np.zeros(1000)
for i in range(1000):
  dist_rec_array[i] = dict[i]['Distance']

end_time = time.time()
print("--- %s secs ---" % (end_time - start_time))
print(dict[dist_rec_array.argmin()])
#%% Print out the best route for SA

best_SA = dict[dist_rec_array.argmin()]
path_city_name = []
path_id = np.insert(best_SA['Path'], [0, len(best_SA['Path'])], [0, 0])

best_route = ' → '.join([city_data[i] for i in path_id])

print("The best route:", best_route, "\nThe total distance:", best_SA['Distance'])

#%% 4. Genetic Algorithm

'''
Pseudocode

START
Generate the initial population
Compute fitness
REPEAT
    Selection
    Crossover
    Mutation
    Compute fitness
UNTIL population has converged
STOP
'''

RejectionSampler = RejectionSampling(threshold=155033, num_samples=1000)
GoodPaths = RejectionSampler.good_path_gen()


start_time = time.time()

class Genetic_algorithm(object):
    def __init__(self, distance_array, iteration, mutation_prop):
        self.distance_array = distance_array
        self.pop_list = None
        self.iteration = iteration
        self.mutation_prop = mutation_prop
        self.population_size = 1000
        self.total_GA = []
    
    def get_dist(self, seq):
        seq0 = np.insert(seq, [0, len(seq)], [0, 0])
        return sum([self.distance_array[c1, c2] for c1, c2 in zip(seq0[:-1], seq0[1:])])
    
    def roulette_wheel_selection(self):
        list_ = self.pop_list.copy()
        selection = []
        for _ in range(2):
            fitness_ls = [self.get_dist(ind) for ind in list_]
            f_sum = sum(fitness_ls)
            probability = [f/f_sum for f in fitness_ls]
            p = np.random.random_sample()
            sum_prob = 0
            for i, prob in enumerate(probability):
                sum_prob += prob
                if sum_prob >= p:
                    target = list_.pop(i)
                    selection.append(target)
                    break
        return selection[0], selection[1]
    
    def uniform_crossover(self, gp_1, gp_2):
        index = int(np.random.choice(len(gp_1), 1))
        new_gp_1, new_gp_2 = gp_1[:index], gp_2[:index]
        ls_1, ls_2 = [], []
        for g1, g2 in zip(gp_1[index:], gp_2[index:]):
            ls_1.append( (int(np.where(gp_2==g1)[0]), g1) )
            ls_2.append( (int(np.where(gp_1==g2)[0]), g2) )
        ls_1, ls_2 = sorted(ls_1), sorted(ls_2)
        ls_1, ls_2 = np.array([g for (i, g) in ls_1]), np.array([g for (i, g) in ls_2])
        new_gp_1 = np.concatenate((new_gp_1, ls_1))
        new_gp_2 = np.concatenate((new_gp_2, ls_2))
        return new_gp_1, new_gp_2
    
    def mutation(self, gp, point=5):
        # mutation for 排列編碼，隨機選取 {point} 個點，將其向左一個位置做交換
        index = np.random.choice(len(gp), 5, replace=False)
        index = np.insert(index, [len(index)], [index[0]])
        new_gp = gp.copy()
        for i, j in zip(index[:-1], index[1:]):
            new_gp[i] = gp[j]
        return new_gp

    def select_candidata(self, seq_list):
        # 產生 population list: 從範圍內挑選 {self.population_size} 個
        if self.population_size < len(seq_list):
            candidata = np.random.choice(range(len(seq_list)), self.population_size)
            self.pop_list = [seq_list[i] for i in candidata]
        else:
            self.pop_list = seq_list
        self.best_y = max([self.get_dist(ind) for ind in self.pop_list])
        self.total_GA = [self.best_y]
        
    def main_program(self, seq_list):
        self.select_candidata(seq_list)
        i = 0
        while i < self.iteration:
            (gp_1, gp_2) = self.roulette_wheel_selection()
            new_gp_1, new_gp_2 = self.uniform_crossover(gp_1, gp_2)
            if np.random.random_sample() < self.mutation_prop:
                new_gp_1 = self.mutation(new_gp_1)
                new_gp_2 = self.mutation(new_gp_2)
            
            candidate = [self.get_dist(gp) for gp in [gp_1, gp_2, new_gp_1, new_gp_2]]
            min_value = [[k, v] for k, v in enumerate(candidate) if v==min(candidate)][0]
            
            if min_value[0] > 1:
                replacement = []
                for ind in self.pop_list:
                    if (ind == gp_1).all():
                        replacement.append(new_gp_1)
                    elif (ind == gp_2).all():
                        replacement.append(new_gp_2)
                    else:
                        replacement.append(ind)
                self.pop_list = replacement
                
                if min_value[1] < self.best_y:
                    if min_value[0] == 2:
                        self.best_x, self.best_y = new_gp_1, min_value[1]
                    elif min_value[0] == 3:
                        self.best_x, self.best_y = new_gp_2, min_value[1]
            i += 1
            self.total_GA.append(self.best_y)

GA = Genetic_algorithm(distance_array=distance_array, iteration=1000, mutation_prop=0.5)
GA.main_program(seq_list = GoodPaths)

end_time = time.time()
print("--- %s secs ---" % (end_time - start_time))
print("The best path:", GA.best_x, "\nThe total distance:", GA.best_y)

#%%

path_id = np.insert(GA.best_x, [0, len(GA.best_x)], [0, 0])
best_route = ' → '.join([city_data[i] for i in path_id])

print("The best route:", best_route, "\nThe total distance:", GA.best_y)


# %%


plt.plot(range(1001), GA.total_GA, label='Genetic Algorithm')
plt.plot(SA_x, SA_y, label='Simulated Annealing')

plt.ylabel("Best distance")
plt.xlabel("Iterations")
plt.legend()
plt.show()

# %%

range(1001)


# %%





# %%




# %%