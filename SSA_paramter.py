# The basic parameters of D-MIMO network
import math
import numpy as np

# The basic parameters of D-MIMO network
# In our setting, N' = N, K' = K.

# the number of transmitter( receiver)
K = 2
# the number of antennas of each node
N = 3
# the number of spatial stream of each node
L_node = [3,2]
# the number of total spatial stream
L = 0
for i in range(K):
    L = L + L_node[i]
# size of finite field
q = 3
# the total energy
E_total = 1e4
N_0 = 1
#SNR = E_total/(N_0*K)

# NC generator matrix
G = np.array([[1, 0, 0, 1, 0],
              [0, 1, 0, 0, 1],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 1, 0],
              [1, 0 ,0, 0, 1]])

# check the feasibility of parameters
def check_param_feasible(N,L):
    K_bar = math.ceil(float(L)/N)
    if L >= (N**2 *K_bar / K_bar * N - N):
        raise Exception('The parameters are unachievable!')
    else:
        pass

# index function I(i,l), represent the index of l-th stream of user i
# The index count from 0!
def index_I(i,l):
    ind = l
    if i > 0:
        for ii in range(i):
            ind = L_node[ii] + ind
    else:
        pass
    return ind

# index inverse function I^{-1}(\tau),
# represent which user the stream with the index belongs
# The index count from 0!
def index_inv(ind):
    i = 0
    ind = ind + 1
    sum_L_node = L_node[i]
    while ind > sum_L_node:
        i = i + 1
        sum_L_node = sum_L_node + L_node[i]
        if ind <= sum_L_node:
            break
        else:
            pass
    return i

# the index of non-zero element of g_{j,l'} ('support element')
# the input l is the index of BS
def index_S(j, l):
    S = []
    g = G[index_I(j,l)]
    for ii in range(len(g)):
        if g[ii] != 0:
            S.append(ii)
    return S

# the index of stream at BS j which consists of the stream I(i,l)
def index_eta(i,l,j):
    eta = None # the column are all zeros
    for ll in range(L_node[j]): # ll is the index of stream in BS j
        if G[index_I(j,ll)][index_I(i,l)] != 0:
            eta = ll
            break

    return eta


if __name__ == '__main__':
    print 'Go!'
    check_param_feasible(N,L)
    print 'The NC generator matrix:\n', G
    print index_I(1,1)
    print index_inv(3)
    print index_S(1,1)
    print index_eta(0,1,1)


