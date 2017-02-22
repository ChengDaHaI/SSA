#!/usr/bin/python2.7
# coding=utf-8
#*************************
# Generate all feasible G matrix, which is with shape (K * M, L)
# Date   : 2017-2-22
# Author : Hai Cheng
# Email  : chenghai@shanghaitech.edu.cn
# *************************

import os, time
import matplotlib.pyplot as pyplot
from multiprocessing import Pool
from scipy import optimize
from SSA_paramter import *
from functools import reduce



# Function: Generate all feasible G_l matrix, which is with shape (M,L).
#           Besides, we now only consider the case of each elements being 1.
# Input   : M, the number of antenna in the BS; K, the number of users; L_node, list of the number of streams in each users
# Output  : all feasible G_l matrix(stacked row by row) in a BS and the number of G_l

def generate_G_l(M = M, K = K, L_node = L_node):

    L = sum(L_node)
    G_l = np.zeros((M, L))

    # the number of possible block matrix in each ``block''
    block_num_list = []
    for i in range(1,K):
        block_num = 1
        for l in range(M-L_node[i]+1,M+1):
            block_num = block_num * l
        block_num_list.append(block_num)

    # store all possible G_l, where reduce(lambda x,y:x*y,block_num_list)
    # return the cumulatively product of elements in block_num_list
    num_G_l = reduce(lambda x,y:x*y,block_num_list)
    G_l_full = np.zeros((M * num_G_l, L))

    # fix the elements in the first block of G_l
    for l in range(min(M,L_node[0])):
        G_l[l,l] = 1

    # block_matrix_list_1 stores all feasible block matrix in block 1
    block_matrix_list_1 = np.zeros((M * block_num_list[0], L_node[1]))
    # generate and store each possible block matrices
    for ii in range(block_num_list[0]): # block 1
        block_matrix = np.zeros((M, L_node[1]))
        ind = range(M)
        ind2 = range(L_node[1])
        block_matrix[ii % M, ind2[0]] = 1
        ind.remove(ii % M)
        ind2.pop(0)
        if len(ind2) >= 1:
            block_matrix[list(ind)[ii / M], ind2[0]] = 1
        if (M >= 3) & (L_node[1] >= 3):
            ind.pop(ii / M)
            ind2.pop(0)
            block_matrix[list(ind)[0], ind2[0]] = 1
        block_matrix_list_1[ii * M: (ii + 1) * M] = block_matrix
    if K == 3:
        # block_matrix_list_2 stores all feasible block matrix in block 2
        block_matrix_list_2 = np.zeros((M * block_num_list[1], L_node[2]))
        for ii in range(block_num_list[1]): # block 2
            block_matrix = np.zeros((M, L_node[2]))
            ind = range(M)
            ind2 = range(L_node[2])
            block_matrix[ii % M, ind2[0]] = 1
            ind.remove(ii % M)
            ind2.remove(ind2[0])
            if len(ind2) >= 1:
                block_matrix[list(ind)[ii/M], ind2[0]] = 1
            if (M >= 3) & (L_node[2] >= 3):
                ind.remove(ii/M)
                ind2.remove(ind2[0])
                block_matrix[list(ind)[0], ind2[0]] = 1
            block_matrix_list_2[ii * M: (ii + 1) * M] = block_matrix

    # generate and store each possible G_l matrix with generated block_matrix_list_1 and block_matrix_list_2
    if K == 2:
        for ii in range(block_num_list[0] ):
            G_l_temp = G_l
            G_l_temp[:, L_node[0]:L_node[0] + L_node[1]] = block_matrix_list_1[(ii % block_num_list[0]) * M: (ii % block_num_list[0] + 1) * M, :]
            # G_l_temp[:, L_node[0] + L_node[1]:L] = block_matrix_list_2[
            #                                        (ii % block_num_list[1]) * M: (ii % block_num_list[1] + 1) * M, :]
            G_l_full[ii * M:(ii + 1) * M, :] = G_l_temp
    if K == 3:
        for ii in range(block_num_list[0] * block_num_list[1]):
            G_l_temp = G_l
            G_l_temp[:, L_node[0]:L_node[0] + L_node[1]] = block_matrix_list_1[(ii / block_num_list[1]) * M : (ii / block_num_list[1] + 1) * M , :]
            G_l_temp[:, L_node[0] + L_node[1]:L] = block_matrix_list_2[(ii % block_num_list[1]) * M: (ii % block_num_list[1] + 1) * M, :]
            G_l_full[ii * M:(ii+1) * M, :] = G_l_temp

    return G_l_full, num_G_l

# Function: Output a possible (complete) G matrix with shape(K*M, K*N) from the input -- a index.
#           The range of ind_G is from 0 to 36 * 36 * 36 -1 (in the case of 3*3 of our paper)
# Input   :
# Output  :

def generate_G_complete(ind_G, G_l_full, num_G_l, M, K, L):

    if ind_G >= num_G_l ** K:
        raise Exception('Wrong index of G! Exceed the range!')
    G_comp = np.zeros((K * M, L))

    # ind0 is the index of G_l in G_l_full of BS1. The remaining follows.

    if K ==2:
        ind0 = ind_G / num_G_l
        ind1 = ind_G % num_G_l
        if (ind0 != ind1):
            G_comp[0 * M: 1 * M, :] = G_l_full[ind0 * M: (ind0 + 1) * M, :]
            G_comp[1 * M: 2 * M, :] = G_l_full[ind1 * M: (ind1 + 1) * M, :]
        else:
            # if two indices equal, that implies the matrix is not full rank
            # print 'G_comp w.r.t ind_G is rank-deficient!'
            return False
    elif K == 3:
        ind0 = ind_G / (num_G_l ** (K - 1))
        ind1 = (ind_G / num_G_l) % num_G_l
        ind2 = ind_G % num_G_l
        if (ind0 != ind1) & (ind0 != ind2) & (ind1 != ind2):
            G_comp[0 * M: 1 * M, :] = G_l_full[ind0 * M: (ind0 + 1) * M, :]
            G_comp[1 * M: 2 * M, :] = G_l_full[ind1 * M: (ind1 + 1) * M, :]
            G_comp[2 * M: 3 * M, :] = G_l_full[ind2 * M: (ind2 + 1) * M, :]
        else:
            # if two indices equal, that implies the matrix is not full rank
            # print 'G_comp w.r.t ind_G is rank-deficient!'
            return False
    else:
        raise Exception("Wrong K !")
    # G_comp[2 * M: 3 * M, :] = G_l_full[ind2 * M: (ind2 + 1) * M, :]
    # print 'complete G matrix :\n', G_comp
    rank_G_comp = np.linalg.matrix_rank(G_comp)
    # print 'its rank is:\n', rank_G_comp
    if rank_G_comp == min(K * M, L):
        return G_comp
    else:
        # print 'G_comp w.r.t ind_G is rank-deficient!'
        return False


if __name__ == "__main__":

    G_l_full, num_G_l = generate_G_l(M, K, L_node)
    # print 'all possible G_l matrix:\n', res
    if K == 3:
        ind_G = 36 * 36 * 10 + 23
    if K ==2:
        ind_G = 6 * 2 + 4
    num_fea_G = 0
    for ind_G in range(num_G_l ** K):
        G_comp = generate_G_complete(ind_G, G_l_full, num_G_l, M, K, L)
        if type(G_comp) != bool:
            num_fea_G += 1
    print 'the number of full column rank complete G is:\n', num_fea_G
