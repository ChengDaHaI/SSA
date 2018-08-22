#!/usr/bin/python2.7
# coding=utf-8
#*************************
# Generate channel coefficient matrix with pass loss fading and spatial correlation
# Date   : 2018-6-28
# Author : Hai Cheng
# Email  : chenghai@shanghaitech.edu.cn
# *************************
import numpy as np
import scipy
from scipy import integrate



'''
sptail correlation matrix caiph's conference paper
'''
# Function: generate the spatial correlation matrix
# def spaCorr(M, theta, delta, D):
#
#     R = np.zeros((M, M))
#     for m in range(M):
#         for p in range(M):
#             # f = lambda x : 1/(2*delta) * np.exp( 2*np.pi*D* np.abs(m - p) * np.sin(x) )
#             f = lambda x: 1 / (2 * delta) * np.cos(2 * np.pi * D * (m - p) * np.sin(x))
#             res = integrate.quad(f, theta - delta, theta + delta)
#             R[m, p] = res[0]
#     return R
#
# # Function: generate the channel coefficient matrix H_{j, k}
# def subchanMatrix(M, N, delta, D, d):
# #
#     H_tilde   = np.random.randn(M, N)
#     theta_r = np.random.rand(1)[0] / 2 * np.pi
#     theta_t = np.random.rand(1)[0] / 2 * np.pi
#     R_receive = spaCorr(M, theta_r, delta, D)
#     R_transmit= spaCorr(N, theta_t, delta, D)
#
#     H_jk = np.dot( np.dot(np.sqrt(R_receive), H_tilde), np.transpose( np.sqrt(R_transmit ))) / d**(1.5)
#     '''
#     testing topliez matrix
#     '''
#     # H_jk = H_tilde / d**(1.5)
#     # H_jk = np.dot( np.dot(np.sqrt(scipy.linalg.toeplitz(0.2 * np.array(range(1,M+1)) )), H_tilde), np.transpose( np.sqrt(scipy.linalg.toeplitz(0.3 * np.array(range(1,M+1)) ) ))) / d**(1.5)
#     return H_jk




'''
sptail correlation matrix in (34) in Junjie Ma's Paper
'''

# Function: generate spatial correlation matrix, with $alpha \in [0, 1) $ being the correlation coefficient.
def spaCorr(M, alpha):
    R = np.zeros((M, M))
    for m in range(M):
        for n in range(M):
            res = np.power(alpha, np.abs(m - n))
            R[m, n] = res
    return R

def subchanMatrix(M, N, alpha_min, alpha_max, d):
    H_tilde   = np.random.randn(M, N)
    alpha_r = np.random.uniform(alpha_min, alpha_max)
    alpha_t = np.random.uniform(alpha_min, alpha_max)
    R_receive = spaCorr(M, alpha_r)
    R_transmit= spaCorr(N, alpha_t)

    H_jk = np.dot( np.dot(np.sqrt(R_receive), H_tilde), np.transpose( np.sqrt(R_transmit ))) / d**(1.5)
    '''
    testing topliez matrix
    '''
    # H_jk = H_tilde / d**(1.5)
    # H_jk = np.dot( np.dot(np.sqrt(scipy.linalg.toeplitz(0.2 * np.array(range(1,M+1)) )), H_tilde), np.transpose( np.sqrt(scipy.linalg.toeplitz(0.2 * np.array(range(1,M+1)) ) ))) / d**(1.5)
    return H_jk

def chanMatrix(M, N, J, K):

    # simulation parameters
    alpha_min = 0.1
    alpha_max = 0.2
    d_min = 10
    d_max = 100
    # channel matrix
    H = np.zeros((K*M, K*N))

    for j in range(J):
        for k in range(K):

            d     = np.random.uniform(d_min, d_max)
            H_jk  = subchanMatrix(M, N, alpha_min, alpha_max, d)
            # H_jk = subchanMatrix(M, N, D, d)
            H[j * M:(j + 1) * M][:, k * N:(k + 1) * N] = H_jk

    return H

if __name__ == "__main__":

    M = 3
    N = 3
    K = 2
    J = 2
    theta = np.random.rand(1)[0] /2 * np.pi
    delta = 10.0 / 180 * np.pi
    D = 1.0/2*3e8/2e9
    d = 10

    # print spaCorr(M, theta, delta, D)
    #
    # print subchanMatrix(M, N, delta, D, d)
    #
    # print chanMatrix(M, N, J, K)
    #
    # print np.random.randn(K*M, K*N)

    Power_path = 0
    Power_gaus = 0
    for i in range(100):
        H_path = chanMatrix(M, N, J, K)
        H_gaus = np.random.randn(K*M, K*N)
        Power_path = Power_path + np.linalg.norm(H_path.reshape((K*J*M*N, 1)))
        Power_gaus = Power_gaus + np.linalg.norm(H_gaus.reshape((K*J*M*N, 1)))
    print 'path fading:', Power_path, 'rayleigh fading:', Power_gaus