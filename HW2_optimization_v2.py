#%%
import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
from numpy.random import choice as np_choice
from itertools import combinations
from sys import maxsize

#%% 2.B 

getbinary = lambda x, n: format(x, 'b').zfill(n)

weight_list = [3.3, 3.4, 6.0, 26.1, 37.6, 62.5, 100.2, 141.1, 119.2, 122.4, 247.6, 352, 24.2, 32.1, 42.5]
points_list = [7, 8, 13, 29, 48, 99, 177, 213, 202, 210, 380, 485, 9, 12, 15]
all_set = 2**15
possible_set = []
impossible_set = []
for i in range(all_set):
    w = 0
    p = 0
    two_all = getbinary(i, 15)
    if int(two_all[0]) + int(two_all[1]) + int(two_all[2]) == 0:
        impossible_set.append(two_all)
        continue
    elif int(two_all[3]) + int(two_all[4]) + int(two_all[5]) == 0:
        impossible_set.append(two_all)
        continue
    elif int(two_all[12]) + int(two_all[13]) + int(two_all[14]) == 0:
        impossible_set.append(two_all)
        continue
    else:
        for j in range(15):
            if two_all[j] == '1':
                w += weight_list[j]
                p += points_list[j]
        if w > 529:
            impossible_set.append(two_all)
        else:
            possible_set.append(two_all)

print('The number of possible combinations:',len(possible_set))


#%% 2.C 
def point_count(init_set):
    point = 0
    for i in range(len(init_set)):
        if init_set[i] == '1':
            point += points_list[i]
    if init_set[0] == 1 and init_set[5] == 1:
        point += 5
    if init_set[3] == 1 and init_set[8] == 1:
        point += 15
    elif init_set[3] == 1 and init_set[9] == 1:
        point += 15
    if init_set[7] == 1 and init_set[5] == 1 and init_set[14] == 1:
        point += 25
    elif init_set[10] == 1 and init_set[5]  == 1 and init_set[14] == 1:
        point += 25
    if init_set[12] == 1 and init_set[13]  == 1 and init_set[14] == 1:
        point += 70
    return point


def check_possible(new_set):
    new_str = ''.join(new_set)
    #print(type(new_str))
    if new_str in impossible_set:
        return False
    else:
        return True


def wheel(init_populate):
    point_gene = []
    for i in range(len(init_populate)):
        point_gene.append(point_count(init_populate[i]))
    a = random.randint(1, sum(point_gene))
    point_sum = 0
    k = 0
    while a > point_sum:
        point_sum += point_gene[k]
        k += 1
    #print(k)
    return init_populate[k - 1]

def mutation(father, mother):
    length = 3
    s = random.randint(0, 14)
    for i in range(length):
        if father[(i + s) % 15] == '1':
            father[(i + s) % 15] = '0'
        elif father[(i + s) % 15] == '0':
            father[(i + s) % 15] = '1'
        if mother[(i + s) % 15] == '1':
            mother[(i + s) % 15] = '0'
        elif mother[(i + s) % 15] == '0':
            mother[(i + s) % 15] = '1'
    return father, mother


def cross(father, mother):
    rate = 0.1
    father = list(father)
    mother = list(mother)
    for i in range(len(father)):
        r = random.random()
        if r <= rate:
            c = father[i]
            father[i] = mother[i]
            mother[i] = c
    father, mother = mutation(father, mother)
    return father, mother


def compare(init_populate):
    popu_point = []
    for i in range(len(init_populate)):
        popu_point.append(point_count(init_populate[i]))
    for j in range(2):
        a = popu_point.index(min(popu_point))
        popu_point.pop(a)
        init_populate.pop(a)
    max_point = max(popu_point)
    return init_populate, max_point


def gene(steps, size, possible_set):
    #random.seed(49)
    popu_point = []
    max_point_list = []
    x_gene = []
    init_populate = random.choices(possible_set, k = size)
    for i in range(len(init_populate)):
        popu_point.append(point_count(init_populate[i]))
    max_point_list.append(max(popu_point))
    for i in range(steps):
        father = wheel(init_populate)
        init_populate.remove(father) #扣掉已抽的 set 再做第二次 wheel
        mother = wheel(init_populate)
        init_populate.append(father) #把第一次的 set 加回來
        a = False
        while a == False:
            son, daughter = cross(father, mother)
            a = check_possible(son) and check_possible(daughter)
        init_populate.append(son)
        init_populate.append(daughter)
        init_populate, max_point = compare(init_populate)
        max_point_list.append(max_point)
        x_gene.append(i)
    return init_populate, max_point_list, x_gene

#%% 2.D GA Optimization

steps = 20
size = 10
populate, max_point_list, x_gene = gene(steps, size, possible_set)
x_gene.append(20)
#print(x_gene)
#print(max_point_list)
plt.plot(x_gene, max_point_list)
plt.xlabel('times')
plt.ylabel('points')
plt.show()
print('Gene Algorithm:',max(max_point_list))


#%% 2.E HILL CLIMBING 
def mountain_move(ori_point, init_set):
    new_set = list(getbinary(0, 15))
    while check_possible(new_set) == False:
        new_set = init_set
        change_unit = random.randint(0, 14)
        if new_set[change_unit] == '1':
            new_set[change_unit] = '0'
        elif new_set[change_unit] == '0':
            new_set[change_unit] = '1'
    new_point = point_count(new_set)
    
    if new_point > ori_point:
        init_set = new_set
        ori_point = new_point
    else:
        new_set = init_set
    return init_set, ori_point


#random.seed(42)
init_num = 1
y_mount = []
x_mount = []
while getbinary(init_num, 15) in impossible_set:
    init_num = random.randint(1, all_set)
    init_set = getbinary(init_num, 15)
init_set = list(init_set)
ori_point = point_count(init_set)
for i in range(200):
    init_set, ori_point = mountain_move(ori_point, init_set)
    y_mount.append(ori_point)
    x_mount.append(i)
plt.plot(x_mount, y_mount)
plt.xlabel('times')
plt.ylabel('points')
plt.show()
print("best set = ", init_set, "best survival point = ", ori_point)


#%% 2.E RANDOM WALK 

def walk_move(ori_point, init_set):
    new_set = list(getbinary(0, 15))
    while check_possible(new_set) == False:
        new_set = init_set
        change_unit = random.randint(0, 14)
        if new_set[change_unit] == '1':
            new_set[change_unit] = '0'
        elif new_set[change_unit] == '0':
            new_set[change_unit] = '1'
    new_point = point_count(new_set)
    if new_point > ori_point:
        init_set = new_set
        ori_point = new_point
    return init_set, ori_point


#random.seed(42)
init_num = 1
x_walk = []
y_walk = []
while getbinary(init_num, 15) in impossible_set:
    init_num = random.randint(1, all_set)
    init_set = getbinary(init_num, 15)
init_set = list(init_set)
ori_point = point_count(init_set)
for i in range(200):
    init_set, ori_point = walk_move(ori_point, init_set)
    y_walk.append(ori_point)
    x_walk.append(i)
plt.plot(x_walk, y_walk)
plt.xlabel('times')
plt.ylabel('points')
plt.show()
print("best set = ", init_set, "best survival point = ", ori_point)



#%% 2.F GENETIC ALGORITHM


steps = 200
size = 10
populate, max_point_list, x_gene = gene(steps, size, possible_set)
x_gene.append(200)
#print(x_gene)
#print(max_point_list)
plt.plot(x_gene, max_point_list)
plt.xlabel('times')
plt.ylabel('points')
plt.show()
print(max(max_point_list))



#%% Three lines in one plot

plt.plot(x_mount, y_mount, label='Hill Climbing')
plt.plot(x_walk, y_walk, label='Random Walk')
plt.plot(x_gene, max_point_list, label='Gene')
plt.xlabel('times')
plt.ylabel('points')
plt.legend()
plt.show()


#%% 3 Traveling Salesman Problem

np.set_printoptions(linewidth=500, suppress=True)
path = '/Users/wujhejia/Documents/Python/distance.xlsx'
distance_df = pd.read_excel(path, header=0, usecols = "B : P", skiprows = 0)
distance_array = distance_df.to_numpy()
city_dict = {
    0:'Incheon',1:'Seoul',2:'Busan',3:'Daegu',4:'Daejeon'
   ,5:'Gwangju',6:'Suwon-si',7:'Ulsan',8:'Jeonju',9:'Cheongju-si'
   ,10:'Changwon',11:'Jeju-si',12:'Chuncheon',13:'Hongsung',14:'Muan'
}

#%% 3.C RANDOM WALK 

def random_path(graph):
  N = 14
  path = []
  path.append(0)
  cities_No = list(range(len(graph)))
  # print(len(cities_No)) #14

  for i in range(N):
    randval = random.randint(1, len(cities_No)-1)
    randomCity = cities_No[randval]
    path.append(randomCity)
    cities_No.remove(randomCity)

  return path

def path_distance(graph, path):
  N = 15
  distance = 0
  for i in range(N):
    distance += graph[path[i-1]][path[i]]
  return distance


def Random_walk():
  iterations = 100
  min_dis = maxsize
  # PLOT: define x_list, y_list
  x_list = []
  y_list = []
  for i in range(iterations):
    path = random_path(distance_array)  
    dis = path_distance(distance_array, path)
    if(dis < min_dis):  min_dis = dis

    # PLOT: append i to x_list/ min_dis to y_list
    x_list.append(i+1)
    y_list.append(min_dis)
  return min_dis, x_list, y_list


res_dis, RA_x, RA_y = Random_walk()
print("Random Walk optimal: ", res_dis)
print("RA_x: ", RA_x)
print("RA_y: ", RA_y)

# plt.plot(RA_x, RA_y, label='Random Walk')
# plt.ylabel("Best distance")
# plt.xlabel("Iterations")
# plt.legend()
# plt.show()

#%% 3.D HILL CLIMBING

class HillClimb(object):
    def __init__(self, iter_ceil):
        """
        args:
            n_iteration (int): Number of iterations
            precision (float): Difference of pnew_y and pb_y
            pby_ceil (int): The best pb_y on the record
        """
        self.iter_ceil = iter_ceil
    
    def getNeighbours(self, solution):
        neighbours = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                neighbour = solution.copy()
                neighbour[i] = solution[j]
                neighbour[j] = solution[i]
                neighbours.append(neighbour)
        return neighbours
    
    def objective(self,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14):
        dist = np.array([distance_array[0,s1],distance_array[s1,s2],distance_array[s2,s3],
                         distance_array[s3,s4],distance_array[s4,s5],distance_array[s5,s6],
                        distance_array[s6,s7],distance_array[s7,s8],distance_array[s8,s9],
                        distance_array[s9,s10],distance_array[s10,s11],distance_array[s11,s12],
                        distance_array[s12,s13],distance_array[s13,s14],distance_array[s14,0]
                    ])
        out = np.sum(dist)
        return out

    def run(self):
        times = 1
        pby_rec = []
        x_list = []
        y_list = []
        s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14 = random.sample(range(1,15),14)
        pb_x = np.array([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14])
        pb_y = self.objective(pb_x[0],pb_x[1],pb_x[2],pb_x[3],pb_x[4],pb_x[5],pb_x[6],
                         pb_x[7],pb_x[8],pb_x[9],pb_x[10],pb_x[11],pb_x[12],pb_x[13])
        while times <= self.iter_ceil:
            pbx_neighbors = self.getNeighbours(pb_x)
            x_list.append(times)
            for i in range(len(pbx_neighbors)):
                n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14 = pbx_neighbors[i]
                pnew_x = np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14])
                pnew_y = self.objective(pnew_x[0],pnew_x[1],pnew_x[2],pnew_x[3],pnew_x[4],pnew_x[5],pnew_x[6],
                                    pnew_x[7],pnew_x[8],pnew_x[9],pnew_x[10],pnew_x[11],pnew_x[12],pnew_x[13])
                if pnew_y <= pb_y:
                    pb_x = pnew_x
                    pb_y = pnew_y
                else:
                    pb_x = pb_x
                    pb_y = pb_y

            if times > self.iter_ceil:
                print('Cannot converge. Iteration:', times)
                break
            else:
                times = times+1
                pby_rec.append(pb_y)

        return pb_x, pb_y, x_list, pby_rec

hillclimber = HillClimb(iter_ceil=100)
res_path, res_dis, Hill_x, Hill_y = hillclimber.run()
print("Hill-climbing path: ", res_path)
print("Hill-climbing optimal: ", res_dis, "km")
print("Hill_x: ", Hill_x)
print("Hill_y: ", Hill_y)

#%% 3.E Tabu Search

def random_path(graph):
  N = 14
  path = []
  path.append(0)
  cities_No = list(range(len(graph)))
  # print(len(cities_No)) #14

  for i in range(N):
    randval = random.randint(1, len(cities_No)-1)
    randomCity = cities_No[randval]
    path.append(randomCity)
    cities_No.remove(randomCity)

  return path

def path_distance(graph, path):
  N = 15
  distance = 0
  for i in range(N):
    distance = distance + graph[path[i-1],path[i]]
  
  return distance


def getTabuList(currentPath):
        '''
        Returns a dict of tabu attributes(pair of jobs that are swapped) as keys and [visit_idx, distance]
        Only record the pair to be swapped, "not really" swapped.
        '''
        dict = {}
        for swap in combinations(currentPath, 2):
          if swap[0] != 0 and swap[1] != 0:
            dict[swap] = {"visit_idx": 0, "distance": 0}
        return dict

def Swap(currentPath, pair):
  swapped_path = currentPath.copy()
  idx_i = swapped_path.index(pair[0])
  idx_j = swapped_path.index(pair[1])
  swapped_path[idx_i], swapped_path[idx_j] = swapped_path[idx_j], swapped_path[idx_i]
  return swapped_path


def Tabu(graph):
  iterations = 100
  tabu_tenure = 10
  tabu_list = [] # record the swap-pair in tabu list

  # initialize the best point
  bestPath = random_path(distance_array)
  bestDistance = path_distance(distance_array, bestPath)
  # initialize the starting point
  currentPath = random_path(distance_array) 
  currentDistance = path_distance(distance_array, currentPath)

  # PLOT: define x_list, y_list
  x_list = []
  y_list = []
  
  iter = 0
  iter_ = 1
  while iter < iterations:
    # Set tabu list
    tabu_list = getTabuList(currentPath)
    for pair in tabu_list:
      runPath = Swap(currentPath, pair)
      runDistance = path_distance(distance_array, runPath) # ptest.y
      tabu_list[pair]["distance"] = runDistance

    while True:
      # Check acceptable cases
      tabu_best_path = min(tabu_list, key =lambda x: tabu_list[x]["distance"]) 
      path_dis = tabu_list[tabu_best_path]["distance"] # the minimum distance in all the neighbors.
      visit_idx = tabu_list[tabu_best_path]["visit_idx"]

      if visit_idx < iter_:
        # Start to move
        currentPath = Swap(currentPath, tabu_best_path) 
        currentDistance = path_distance(distance_array, currentPath)
        
        if path_dis < bestDistance:
          bestPath = currentPath
          bestDistance = currentDistance
        
        tabu_list[tabu_best_path]["visit_idx"] = tabu_tenure + iter_
        iter += 1
        iter_ += 1
        
        break

      # Update tabu list (already in tabu list)
      else:
        if path_dis < bestDistance:
          # start to move
          currentPath = Swap(currentPath, tabu_best_path) 
          currentDistance = path_distance(currentPath)
          # print("cur Path: ",currentPath)
          bestPath = currentPath
          bestDistance = currentDistance

          iter_ += 1
          
          break
        else: 
          tabu_list[tabu_best_path]["distance"] = maxsize 
          continue

    # PLOT: append i to x_list/ bestDistance to y_list
    x_list.append(iter)
    y_list.append(bestDistance)

  return bestPath, bestDistance, x_list, y_list


res_path, res_dis, Tabu_x, Tabu_y = Tabu(distance_array)
print("Tabu Search path: ", res_path)
print("Tabu Search optimal: ", res_dis, "km")
print("Tabu_x: ", Tabu_x)
print("Tabu_y: ", Tabu_y)



#%% 3.F Simulated Annealing

class SimAnn(object):
    def __init__(self, n_iterations):
        """
        args:
            n_iteration (int): Number of iterations
            precision (float): Difference of pnew_y and pb_y
            pby_alltime (int): The best pb_y on the record
        """
        self.n_iterations = n_iterations

    def objective(self, s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14):
        dist = np.array([distance_array[0,s1],distance_array[s1,s2],distance_array[s2,s3],
                        distance_array[s3,s4],distance_array[s4,s5],distance_array[s5,s6],
                        distance_array[s6,s7],distance_array[s7,s8],distance_array[s8,s9],
                        distance_array[s9,s10],distance_array[s10,s11],distance_array[s11,s12],
                        distance_array[s12,s13],distance_array[s13,s14],distance_array[s14,0]
                        ])
        out = np.sum(dist)
        return out
    
    def getNeighbours(self, solution):
        neighbours = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                neighbour = solution.copy()
                neighbour[i] = solution[j]
                neighbour[j] = solution[i]
                neighbours.append(neighbour)
        return neighbours

    def getTemp(self, t, temp):
        out = (1 - t/(self.n_iterations))*temp
        return out

    
    def run(self, e=1e-30):
        t = 0
        T0 = 100

        s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14 = random.sample(range(1,15),14)
        pcur_x = np.array([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14])
        pcur_y = self.objective(pcur_x[0],pcur_x[1],pcur_x[2],pcur_x[3],pcur_x[4],pcur_x[5],pcur_x[6],
                            pcur_x[7],pcur_x[8],pcur_x[9],pcur_x[10],pcur_x[11],pcur_x[12],pcur_x[13])
        pb_x, pb_y = pcur_x, pcur_y

        x_list = []
        y_list = []

        while t < self.n_iterations:
            pcurx_neighbors = self.getNeighbours(pcur_x)
            neighbor_idx = random.randint(0,len(pcurx_neighbors)-1)
            pnew_x = pcurx_neighbors[neighbor_idx]
            pnew_y = self.objective(pnew_x[0],pnew_x[1],pnew_x[2],pnew_x[3],pnew_x[4],pnew_x[5],pnew_x[6],
                                pnew_x[7],pnew_x[8],pnew_x[9],pnew_x[10],pnew_x[11],pnew_x[12],pnew_x[13])
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


SimAnneal = SimAnn(n_iterations = 100)   


res_x, res_y, SA_x, SA_y = SimAnneal.run()
print("Simulated Annealing path: ", res_x)
print("Simulated Annealing optimal: ", res_y, "km")
print("SA_x: ", SA_x)
print("SA_y: ", SA_y)

#%% 3.G Ant Colony Optimization


for i in range(15):
    distance_array[i,i] = np.inf


class AntColony(object):
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        Example:
            ant_colony = AntColony(distance_array, n_ants=100, n_best=20, n_iterations=2000, decay=0.95, pby_alltime = 1332)          
        """
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0/dist) ** self.beta)
        norm_row = row/row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], distance_array[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))   
        return path

    def gen_path_dist(self, path):
        total_dist = 0
        for i in path:
            total_dist = total_dist + self.distances[i]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def spread_pheronome(self, all_paths, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:self.n_best]:
            for move in path:
                self.pheromone[move] = self.pheromone[move] +  (1.0/self.distances[move])

    def main(self):
        shortest_path = []
        all_time_shortest_path = ("placeholder", np.inf)
        x_list = []
        y_list = []
        t = 0
        while t < self.n_iterations:
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])

            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path 
                t = t+1    
                self.pheromone = self.pheromone * self.decay 
            else:
                t = t+1       
                self.pheromone = self.pheromone * self.decay  

            # PLOT: append t to x_list/ all_time_shortest_path[1] to y_list
            x_list.append(t)
            y_list.append(all_time_shortest_path[1])    

        return all_time_shortest_path[0],  all_time_shortest_path[1], x_list, y_list


antcolony = AntColony(distance_array, n_ants=30, n_best=5, n_iterations=100, decay=0.9)  
res_path, res_dis, Ant_x, Ant_y = antcolony.main()
print("Ant Colony path: ", res_path)
print("Ant Colony optimal: ", res_dis, "km")
print("Ant_x: ", Ant_x)
print("Ant_y: ", Ant_y)



#%% Ant Colony without class
start = 0
def pick_move(pheromone, dist, visited, alpha=1, beta=1):
        all_inds = range(len(distance_array))
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** alpha * ((1.0/dist) ** beta)
        norm_row = row / row.sum()
        move = np_choice(all_inds, 1, p=norm_row)[0]
        return move


def genPath(num_of_ants, pheromone):
  path_list = []
  for i in range(num_of_ants):
      # path = gen_path(0)
      path = []

      visited = set()
      visited.add(0) # start from 0
      prev = 0
      for i in range(len(distance_array) - 1): 
          move = pick_move(pheromone[prev], distance_array[prev], visited)
          path.append((prev, move))
          prev = move
          visited.add(move)
      path.append((prev, start)) # a new path has been created.

      # Calculate the distance of path
      total_dist = 0
      for i in path:
          total_dist += distance_array[i]
      # add paths to the path list
      path_list.append((path, total_dist)) 

  return path_list


def spread_pheronome(pheromone, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                pheromone[move] = pheromone[move] +  (1.0/distance_array[move])


def Ant(graph):
  for i in range(15):
    distance_array[i,i] = np.inf
  
  pheromone = np.ones(distance_array.shape)/len(distance_array)
  iterations = 100
  num_of_ants = 30
  n_best = 3
  decay = 0.9
  all_time_shortest_path = ("placeholder", np.inf)
  shortest_path = []
  
  # PLOT: define x_list, y_list
  x_list = []
  y_list = []

  t = 0
  while t < iterations:
      all_paths = genPath(num_of_ants, pheromone)
      spread_pheronome(pheromone, all_paths, n_best)
      shortest_path = min(all_paths, key=lambda x: x[1])
      
      if shortest_path[1] < all_time_shortest_path[1]:
          all_time_shortest_path = shortest_path 
          t += 1   
          pheromone *= decay 
      else:
          t += 1       
          pheromone *= decay   

      # PLOT: append t to x_list/ all_time_shortest_path[1] to y_list
      x_list.append(t)
      y_list.append(all_time_shortest_path[1]) 

  return all_time_shortest_path[0],  all_time_shortest_path[1], x_list, y_list

res_path, res_dis, Ant_x, Ant_y = Ant(distance_array)

print("Ant Colony path: ", res_path)
print("Ant Colony optimal: ", res_dis, "km")
print("Ant_x: ", Ant_x)
print("Ant_y: ", Ant_y)

#%% 3.H Comparison Plots for Algorithms

# Random Walk
RA_dis, RA_x, RA_y = Random_walk() 
plt.plot(RA_x, RA_y, label='Random Walk')

# Hill Climbing
hillclimber = HillClimb(iter_ceil=100)
res_path, res_dis, Hill_x, Hill_y = hillclimber.run()
plt.plot(Hill_x, Hill_y, label='Hill Climbing')

# Tabu Search
res_path, res_dis, Tabu_x, Tabu_y = Tabu(distance_array)
plt.plot(Tabu_x, Tabu_y, label = 'Tabu Search')

# Simulated Annealing
SimAnneal = SimAnn(n_iterations = 100)   
res_x, res_y, SA_x, SA_y = SimAnneal.run()
plt.plot(SA_x, SA_y, label = 'Simulated Annealing')

# Ant Colony
#res_path, res_dis, Ant_x, Ant_y = Ant(distance_array)
antcolony = AntColony(distance_array, n_ants=30, n_best=5, n_iterations=100, decay=0.9)  
res_path, res_dis, Ant_x, Ant_y = antcolony.main()
plt.plot(Ant_x, Ant_y, label = 'Ant Colony')


plt.ylabel("Best distance")
plt.xlabel("Iterations")
plt.legend()
plt.show()

#%% 4. PSO (Bonus)

def routeLength(tsp, x):
    if x.ndim == 1:
        solution = x.copy();
        pathLength = 0
        for i in range(len(solution)):
            pathLength += tsp[solution[i - 1]][solution[i]]
        return pathLength
    else:
        pop = x.copy()
        obj = np.zeros(pop.shape[0])
        # run over all population and compute the fitness value
        for  j in range(pop.shape[0]):
            solution = pop[j, :]
            pathLength = 0
            for i in range(len(solution)):
                pathLength += tsp[solution[i - 1]][solution[i]]
            obj[j] = pathLength

        return obj


def find_move(best, x_cur):
    # v = best - x_cur
    x = x_cur.copy()
    numofcity = len(best)   # get length
    moveperm = []
    for i in range(numofcity):
        # check if the best[i] equal to x[i]
        idxinx = np.where(x == best[i])[0]
        
        # if not, swap
        if i != idxinx:
            move = [x[i], x[idxinx][0]]
            moveperm.append(move)
            x[idxinx], x[i] = x[i], x[idxinx]
            
    return moveperm

def updatepos(x_cur, moves):
    x = x_cur.copy()
    for i in range(len(moves)):
        move = moves[i]
        idx1 = np.where(x == move[0])[0]
        idx2 = np.where(x == move[1])[0]
        x[idx1], x[idx2] = x[idx2], x[idx1]

    return x

def updatemove(w, cr1, cr2, moves_inertia, moves_loc, moves_global, maxv):
    # v = w*moves_inertia + cr1*moves_loc + cr2*moves_global
    
    # scalar times velocity
    wint = np.floor(w*len(moves_inertia)).astype(int)
    cr1int = np.floor(cr1*len(moves_loc)).astype(int)
    cr2int = np.floor(cr2*len(moves_global)).astype(int)
    # summation three part of velocity
    moves = moves_inertia[0:wint]
    moves.extend(moves_loc[0:cr1int])
    moves.extend(moves_global[0:cr2int])
    
    # avoid velocity too large
    if len(moves) > maxv:
        numofdel = len(moves)-maxv
        idx = np.random.permutation(len(moves))
        for i in range(numofdel):
            moves.pop()
    
    return moves

# max number of step
n = 100

# cognition factor and social factor
c1 = c2 = 1.0 # only can be 1
w = 1.0 # only can be 1
maxv = 15

# determine the population size
pop_size = 20
pop_dim = np.shape(distance_array)[1]


# Initialize x
# x is of size (num of particle)-by-(dim of a point)
x_start = np.zeros((pop_size, pop_dim), dtype=int)
for j in range(pop_size):
    x_start[j, :] = random_path(distance_array)
particle, dim = x_start.shape
# initial velocity by empty list with size of 1-by-pop_size
v = [[] for _ in range(pop_size)]


# locol best & global best
x_loc_best = x_start.copy()
loc_best_obj = routeLength(distance_array, x_start)
x_global_best = x_loc_best[loc_best_obj.argmin(), :]
global_best_obj = loc_best_obj.min()

# initial position and velocity
x_cur = x_start.copy()


# record the global best and local current (not best) objective
fb_record = np.zeros(n+1)
fm_record = np.zeros(n+1)
objs = routeLength(distance_array, x_start)
fb_record[0] = np.min(objs)
fm_record[0] = np.mean(objs)


for i in range(n):
    print('Step: ' + str(i) + ' f best: ' + str(fb_record[i]))
    r1, r2 = np.random.rand(2)
    # run over all particles and update the position
    for j, solution in enumerate(x_cur):
        # find velocity and update velocity
        moves_inertia = v[j]
        move_loc = find_move(x_loc_best[j, :], solution)
        move_global = find_move(x_global_best, solution)
        tempv = updatemove(w, c1*r1, c2*r2, moves_inertia, move_loc, move_global, maxv)
        v[j] = tempv
        # update position
        tempx = updatepos(x_cur[j, :], tempv)
        x_cur[j, :] = tempx
    
    # run over all particles and update the local and global best
    obj = routeLength(distance_array, x_cur)
    for j, solution in enumerate(x_cur):
        # update the local best (min)
        if (obj[j] < loc_best_obj[j]):
            x_loc_best[j, :] = solution.copy()
            loc_best_obj[j] = obj[j].copy()
                
            # update the global best (min)
            if (obj[j] < global_best_obj):
                x_global_best = solution.copy()
                global_best_obj = obj[j].copy()
    
    # check whether converge
    conv = True
    for j in range(1, x_cur.shape[0]):
        if not np.array_equal(x_cur[0, :], x_cur[j, :]):
            conv = False
            break
    if conv:
        print("All particle are at the same position.")
        break
                
    fb_record[i+1] = global_best_obj
    fm_record[i+1] = np.mean(obj)

print('Best solution: ' + str(x_global_best))
print('Best objective: ' + str(routeLength(distance_array, x_global_best)))

#%% 4. Single Plot for PSO


fig = plt.figure()
plt.plot(np.arange(0, n+1), fb_record,'b-o')
plt.ylabel("Best objective value")
plt.xlabel("Step")
plt.show



#%% 4. Multiple Methods in one plot


# Random Walk
RA_dis, RA_x, RA_y = Random_walk() 
plt.plot(RA_x, RA_y, label='Random Walk')

# Hill Climbing
hillclimber = HillClimb(iter_ceil=100)
res_path, res_dis, Hill_x, Hill_y = hillclimber.run()
plt.plot(Hill_x, Hill_y, label='Hill Climbing')

# Tabu Search
res_path, res_dis, Tabu_x, Tabu_y = Tabu(distance_array)
plt.plot(Tabu_x, Tabu_y, label = 'Tabu Search')

# Simulated Annealing
SimAnneal = SimAnn(n_iterations = 100)   
res_x, res_y, SA_x, SA_y = SimAnneal.run()
plt.plot(SA_x, SA_y, label = 'Simulated Annealing')

# Ant Colony
#res_path, res_dis, Ant_x, Ant_y = Ant(distance_array)
antcolony = AntColony(distance_array, n_ants=30, n_best=5, n_iterations=100, decay=0.9)  
res_path, res_dis, Ant_x, Ant_y = antcolony.main()
plt.plot(Ant_x, Ant_y, label = 'Ant Colony')


plt.plot(np.arange(0, n+1), fb_record, label= 'PSO')

plt.ylabel("Best distance")
plt.xlabel("Iterations")
plt.legend()
plt.show()


#%%




#%%