#!/usr/bin/python2.7
# coding=utf-8
#*************************
# Generate all feasible G matrix, which is with shape (K * M, L)
# Date   : 2017-2-22
# Author : Hai Cheng
# Email  : chenghai@shanghaitech.edu.cn
# *************************

import os, time
from SSA_paramter import *
from functools import reduce
import networkx as nx



# Function: Generate all feasible G_l matrix, which is with shape (M,L).
#           Besides, we now only consider the case of each elements being 1.
# Input   : M, the number of antenna in the BS; K, the number of users; L_node, list of the number of streams in each users
# Output  : all feasible G_l matrix(stacked row by row) in a BS and the number of G_l
def generate_G_l():

    # L = sum(L_node)
    G_l = np.zeros((M, L),np.int8)

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
# Input   : the index, ind_G; all feasible block matrix at BS l, G_l_full;
#           number of feasible block matrix, num_G_l
# Output  : complete G matrix associated with ind_G, G_comp; or False if G_comp is rank-deficient.
def generate_G_complete(ind_G, G_l_full, num_G_l):

    if ind_G >= num_G_l ** K:
        raise Exception('Wrong index of G! Exceed the range!')
    G_comp = np.zeros((K * M, L),np.int8)

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


# Function: converte adjacency matrix into a list of cycle basis in the help pf networkx.cycle_basis()
# Input   : biadjacency matrix G
# Output  : cycle basis  list of G
def cycles_in_G(G_adj):
    G_edges = nx.Graph()
    for i in range(L):
        for j in range(L):
            if G_adj[i, j] != 0:
                # the index of source stream node is j and the
                # index of received spatial stream node is i + L
                G_edges.add_edge(i + L, j)
    # find all cycle basis of G matrix
    cycle_node_list = list(nx.cycle_basis(G_edges))
    num_cycle = len(cycle_node_list)
    return cycle_node_list, num_cycle


# Function: For given cycles in the graph of G, calculate the constraints induced by cycles.
# Input   : biadjacency matrix, G; cycle basis list of G, cycle_list
# Output  : number of all beta variables, index of fixed beta variables, and the constraints.
def beta_constraint(G, cycle_list):

    num_cyc = len(cycle_list)
    # calculate the total beta
    num_beta = 0
    for l in range(sum(L_node[1:K])):
        num_beta += sum(G[:,L_node[0] + l])

    # store the beta equation constraints for each cycle,
    # and the product of element of G. Note that g_prod is multiplied
    # in the -----``RHS''----- of equation constraints.
    beta_equa_list_left  = []
    beta_equa_list_right = []
    g_prod_list          = []
    for ind_cyc in range(num_cyc):
        cycle = cycle_list[ind_cyc]
        cyc_len = len(cycle)
        # index of received streams
        t_ind_list = np.where(np.array(cycle) >= L)[0]
        # index of source streams
        p_ind_list = range(cyc_len)
        for i in list(t_ind_list):
            p_ind_list.remove(i)
        # # source stream node
        # p_node = np.array(cycle)[p_ind_list]
        # # received stream node
        # t_node = np.array(cycle)[p_ind_list]

        # we begin to construct the equation constraints
        # beta_left: list to store beta index of LHS
        beta_left  = []
        # beta_right: list to store beta index of RHS
        beta_right = []
        # product of elements of G in the equation constraints
        g_prod = 1
        for ind in list(t_ind_list):
            t   = cycle[ind] - L
            p_l = cycle[ind - 1]
            if ind + 1 == len(cycle):
                #the right index is the first index in cycle
                p_r = cycle[0]
            else:
                p_r = cycle[ind + 1]
            g_prod = g_prod * (G[t,p_l]/G[t,p_r])
            # calculate the index of beta corresponding to t and p_l
            if index_inv(p_l) != 0:
                beta_ind = np.sum( G[0:t,:][:,L_node[0]:L] ) + np.sum(G[t,L_node[0]:p_l])
                # beta_ind = sum(sum(G[0:t,:][:,sum(L_node[0:index_inv(p_l)]):L])) + G[t,sum(L_node[0:index_inv(p_l)]):p_l]
                # Note that beta_ind of p_l is appended in beta_right list!!!
                beta_right.append(beta_ind)
            else:
                pass
            # calculate the index of beta corresponding to t and p_l
            if index_inv(p_r) != 0:
                beta_ind = np.sum(G[0:t, :][:, L_node[0]:L]) + np.sum(
                    G[t, L_node[0]:p_r])
                # beta_ind = sum(sum(G[0:t,:][:,sum(L_node[0:index_inv(p_r)]):L])) + G[t,sum(L_node[0:index_inv(p_r)]):p_r]
                beta_left.append(beta_ind)
            else:
                pass
        beta_equa_list_left.append(beta_left)
        beta_equa_list_right.append(beta_right)
        g_prod_list.append(g_prod)

    # WE are now ready to compute the fixed beta variables
    beta_equa_len = [0] * num_cyc
    # beta_fixed stored in a list
    # the equation to calculate the fixed beta is stored in beta_equa_constr
    #TODO Try to store it in a dict or class!!!
    beta_fixed    = []
    beta_equa_constr = []
    for ind_cyc in range(num_cyc):
            beta_equa_len[ind_cyc] = len(beta_equa_list_left[ind_cyc])
    for ind in range(num_cyc):
        # find the index of first min len equation
        ind_min_len_equa = beta_equa_len.index(min(beta_equa_len))
        # ind_min_len_equa = beta_equa_len[beta_equa_len.index(min(beta_equa_len))]
        # we choose the leftest index in the left side of the equation as the fixed beta.
        if ind == 0:
            beta_fixed.append(beta_equa_list_left[ind_min_len_equa][0])
            beta_equa_list_left[ind_min_len_equa].remove(beta_fixed[ind])
            beta_equa_constr.append( [beta_equa_list_left[ind_min_len_equa], beta_equa_list_right[ind_min_len_equa],
                                      g_prod_list[ind_min_len_equa], 'left'] )
            # beta_fixed[ind,:] = np.array([min(beta_equa_list_left[ind_min_len_equa]), ind_min_len_equa])
        else:
            equa_list = beta_equa_list_left[ind_min_len_equa] + beta_equa_list_right[ind_min_len_equa]
            for ii in list(beta_fixed[0:ind]):
                if ii in equa_list:
                    equa_list.remove(ii)
            beta_fixed.append(equa_list[0])
            if equa_list[0] in beta_equa_list_left[ind_min_len_equa]:
                # the fixed beta is in the left of equation constraint
                beta_equa_list_left[ind_min_len_equa].remove(beta_fixed[ind])
                beta_equa_constr.append([beta_equa_list_left[ind_min_len_equa],beta_equa_list_right[ind_min_len_equa],
                                         g_prod_list[ind_min_len_equa], 'left'])
            elif equa_list[0] in beta_equa_list_right[ind_min_len_equa]:
                # the fixed beta is in the right of equation constraint
                beta_equa_list_right[ind_min_len_equa].remove(beta_fixed[ind])
                beta_equa_constr.append([beta_equa_list_left[ind_min_len_equa],beta_equa_list_right[ind_min_len_equa],
                                         g_prod_list[ind_min_len_equa], 'right'])
            else:
                raise Exception('Wrong index in equa_list!!!')
        # remove the used equation list
        beta_equa_len.pop(ind_min_len_equa)
        beta_equa_list_left.pop(ind_min_len_equa)
        beta_equa_list_right.pop(ind_min_len_equa)
        g_prod_list.pop(ind_min_len_equa)

    # up to now, we have already calculated all data of beta constraints
    return num_beta, beta_fixed, beta_equa_constr




if __name__ == "__main__":

    G_l_full, num_G_l = generate_G_l()
    # print 'all possible G_l matrix:\n', res
    if K == 3:
        ind_G = 36 * 36 * 10 + 23
    if K ==2:
        ind_G = 6 * 2 + 4
    num_fea_G = 0
    for ind_G in range(num_G_l ** K):
        G_comp = generate_G_complete(ind_G, G_l_full, num_G_l)
        if type(G_comp) != bool:
            num_fea_G += 1
    print 'the number of full column rank complete G is:\n', num_fea_G

    cycle_node_list, num_cycle = cycles_in_G(G)
    print cycle_node_list, num_cycle
    print beta_constraint(G, cycle_node_list)

