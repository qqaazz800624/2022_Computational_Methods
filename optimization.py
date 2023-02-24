#%%
import numpy as np
import pandas as pd
import random

#%%

np.set_printoptions(linewidth=500, suppress=True)
path = '/home/u/qqaazz800624/projs/dscomputing/distance.xlsx'
distance_df = pd.read_excel(path, header=0, usecols = "B : P", skiprows = 0)
distance_array = distance_df.to_numpy()

city_dict = {
    0:'Incheon',1:'Seoul',2:'Busan',3:'Daegu',4:'Daejeon'
   ,5:'Gwangju',6:'Suwon-si',7:'Ulsan',8:'Jeonju',9:'Cheongju-si'
   ,10:'Changwon',11:'Jeju-si',12:'Chuncheon',13:'Hongsung',14:'Muan'
}
city_dict


#%% 3.C RANDOM WALK

def objective(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14):
    dist = np.array([distance_array[0,s1],distance_array[s1,s2],distance_array[s2,s3],
                     distance_array[s3,s4],distance_array[s4,s5],distance_array[s5,s6],
                     distance_array[s6,s7],distance_array[s7,s8],distance_array[s8,s9],
                     distance_array[s9,s10],distance_array[s10,s11],distance_array[s11,s12],
                     distance_array[s12,s13],distance_array[s13,s14],distance_array[s14,0]
    ])
    out = np.sum(dist)
    return out


def randomwalk(iter_ceil=100, precision=1e-30, pby_ceil=1420):
    #s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14 = random.sample(range(1,15),14)
    s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14 = np.array([1,13,9,4,8,5,14,11,10,2,7,3,6,12])
    pb_x = np.array([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14])
    pb_y = objective(pb_x[0],pb_x[1],pb_x[2],pb_x[3],pb_x[4],pb_x[5],pb_x[6],
                     pb_x[7],pb_x[8],pb_x[9],pb_x[10],pb_x[11],pb_x[12],pb_x[13])
    pnew_y = pb_y
    times = 1

    while times < iter_ceil:
        n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14 = random.sample(range(1,15),14)
        pnew_x = np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14])
        pnew_y = objective(pnew_x[0],pnew_x[1],pnew_x[2],pnew_x[3],pnew_x[4],pnew_x[5],pnew_x[6],
                           pnew_x[7],pnew_x[8],pnew_x[9],pnew_x[10],pnew_x[11],pnew_x[12],pnew_x[13])
        times = times + 1

        if pnew_y <= pb_y:
            pb_x = pnew_x
            pb_y = pnew_y
        elif times > iter_ceil:
            print('Cannot converge. Iteration:', times)
            break
        elif times <= iter_ceil and pb_y <= pby_ceil:
            print('Iteration:', times,
                  'pb_y:', pb_y, 'pb_x:', pb_x)
            break
        elif np.abs(pnew_y-pb_y)<=precision and pb_y <= pby_ceil:
            print('y converges. Iteration:', times,
                  'pb_y:', pb_y, 'pb_x:', pb_x)
            break
        else: 
            times = times + 1
    return pb_x, pb_y, times

# pb_x, pb_y, times = randomwalk(iter_ceil=1e9, precision=1e-30)
# print('pb_x:', pb_x, 'pb_y:', pb_y, 'Iteration:', times)

#%%


def getNeighbours(solution):
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)
    return neighbours


def hillclimb(iter_ceil=100, precision=1e-30, pby_ceil=1360):
    times = 1
    pby_rec = []
    s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14 = random.sample(range(1,15),14)
    #s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14 = np.array([13,9,4,8,5,14,11,10,2,7,3,12,1,6])
    pb_x = np.array([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14])
    pb_y = objective(pb_x[0],pb_x[1],pb_x[2],pb_x[3],pb_x[4],pb_x[5],pb_x[6],
                         pb_x[7],pb_x[8],pb_x[9],pb_x[10],pb_x[11],pb_x[12],pb_x[13])
    while times <= iter_ceil:
        pbx_neighbors = getNeighbours(pb_x)
        for i in range(len(pbx_neighbors)):
            n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14 = pbx_neighbors[i]
            pnew_x = np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14])
            pnew_y = objective(pnew_x[0],pnew_x[1],pnew_x[2],pnew_x[3],pnew_x[4],pnew_x[5],pnew_x[6],
                               pnew_x[7],pnew_x[8],pnew_x[9],pnew_x[10],pnew_x[11],pnew_x[12],pnew_x[13])
            if pnew_y <= pb_y:
                pb_x = pnew_x
                pb_y = pnew_y
            else:
                pb_x = pb_x
                pb_y = pb_y

        if times > iter_ceil:
            print('Cannot converge. Iteration:', times)
            break
        elif times <= iter_ceil and pb_y <= pby_ceil:
            print('Iteration:', times,
                  'pb_y:', pb_y, 'pb_x:', pb_x)
            break
        elif np.abs(pnew_y-pb_y)<=precision and pb_y <= pby_ceil:
            print('y converges. Iteration:', times,
                  'pb_y:', pb_y, 'pb_x:', pb_x)
            break
        else:
            times = times+1
            pby_rec.append(pb_y)
    return pb_x, pb_y, times, pby_rec

pb_x, pb_y, times, pbx_rec = hillclimb(iter_ceil=1e5, precision=1e-30)
print('pb_x:', pb_x, 'pb_y:', pb_y, 'Iteration:', times)


#%%




#%%