# !/usr/bin/python2.7
# coding=utf-8
# *************************
# interference alignment scheme for comparison in G-SSA-PNC response
# none symbol extension is used
#  np.ones matrix is used as beamforming matrix in users
# IA and full decoding are employed in BSs
# *************************

import numpy as np
from SSA_fading_channel_model import chanMatrix
import matplotlib.pyplot as pyplot
from multiprocessing import Pool
import time


# Function: calculate the sum rate of two user two BS system
# user 1 transmit 2 streams, user 2 transmit 1 stream
def rate_IA_two_user(H, M, N, SNR):

    # beamforming matrix at users
    P_1 = np.ones((N, 2))* np.sqrt(SNR/(2*N))
    P_2 = np.ones((N, 1))* np.sqrt(SNR/N)

    # calculate the zero-forcing vecotr at BSs
    [s, v, d] = np.linalg.svd(np.dot(H[0 * M:(0 + 1) * M][:, 1 * N:(1 + 1) * N], P_2))
    F_1 = np.expand_dims(s[:, -1] / np.linalg.norm(s[:, -1]), axis=0)
    [s, v, d] = np.linalg.svd(np.dot(H[1 * M:(1 + 1) * M][:, 0 * N:(0 + 1) * N], P_1))
    F_2 = np.expand_dims(s[:, -1] / np.linalg.norm(s[:, -1]), axis=0)

    R_1 = 2 * 0.5 * np.log2(1 + (np.squeeze(np.dot(np.dot( F_1, H[0 * M:(0 + 1) * M][:, 0 * N:(0 + 1) * N] ), P_1[:, 0]))/np.linalg.norm(F_1))**2 )
    R_2 =  0.5 * np.log2(1 + (np.squeeze(np.dot(np.dot(F_2, H[1 * M:(1 + 1) * M][:, 1 * N:(1 + 1) * N]), P_2)) / np.linalg.norm(F_2)) ** 2)

    return R_1 + R_2


# Function: calculate the sum rate of three user three BS system
# user 1 transmit 2 streams, user 2 transmit 1 stream, user 3 transmit 1 stream.
def rate_IA_three_user(H, M, N, SNR):

    # beamforming matrix at users
    # following the derivation in Jafar's IA paper, Fig.1, page 4.
    P_2 = np.ones((N, 1))
    P_2 = P_2* np.sqrt(SNR) / np.linalg.norm(P_2)
    P_3 = np.dot(np.linalg.pinv(H[0 * M:(0 + 1) * M][:, 2 * N:(2 + 1) * N]), np.dot( H[0 * M:(0 + 1) * M][:, 1 * N:(1 + 1) * N], P_2))
    P_3 = P_3 * np.sqrt(SNR) / np.linalg.norm(P_3)
    P_11 = np.dot(np.linalg.pinv(H[1 * M:(1 + 1) * M][:, 0 * N:(0 + 1) * N] ), H[1 * M:(1 + 1) * M][:, 2 * N:(2 + 1) * N] )
    P_11 = np.dot(P_11, P_3)
    P_12 = np.dot(np.linalg.pinv(H[2 * M:(2 + 1) * M][:, 0 * N:(0 + 1) * N] ), H[2 * M:(2 + 1) * M][:,1 * N:(1 + 1) * N] )
    P_12 = np.dot(P_12, P_2)
    P_1 = np.ones((N, 2))
    P_1[:, 0] = np.squeeze(P_11, axis=1)
    P_1[:, 1] = np.squeeze(P_12, axis=1)
    P_1 = P_1 * np.sqrt(SNR) / np.linalg.norm(P_1)

    # calculate the zero-forcing vecotr at BSs
    H_tilde_1   = np.concatenate((np.dot(H[0 * M:(0 + 1) * M][:, 1 * N:(1 + 1) * N], P_2 ),
                                  np.dot(H[0 * M:(0 + 1) * M][:, 2 * N:(2 + 1) * N], P_3 )), axis = 1 )
    [s, v, d] = np.linalg.svd(H_tilde_1)
    F_1 = np.expand_dims(s[:, -1] / np.linalg.norm(s[:, -1]), axis=0)
    H_tilde_2   = np.concatenate((np.dot(H[1 * M:(1 + 1) * M][:, 0 * N:(0 + 1) * N], P_1 ),
                                  np.dot(H[1 * M:(1 + 1) * M][:, 2 * N:(2 + 1) * N], P_3 )), axis = 1 )
    [s, v, d] = np.linalg.svd(H_tilde_2)
    F_2 = np.expand_dims(s[:, -1] / np.linalg.norm(s[:, -1]), axis=0)
    H_tilde_3 = np.concatenate((np.dot(H[2 * M:(2 + 1) * M][:, 0 * N:(0 + 1) * N], P_1),
                                np.dot(H[2 * M:(2 + 1) * M][:, 1 * N:(1 + 1) * N], P_2)), axis=1)
    [s, v, d] = np.linalg.svd(H_tilde_3)
    F_3 = np.expand_dims(s[:, -1] / np.linalg.norm(s[:, -1]), axis=0)
    R_1_1 =  0.5 * np.log2(1 + (np.squeeze(np.dot(np.dot( F_1, H[0 * M:(0 + 1) * M][:, 0 * N:(0 + 1) * N] ), P_1[:, 0]))/np.linalg.norm(F_1))**2 )
    R_1_2 = 0.5 * np.log2(1 + (np.squeeze(np.dot(np.dot(F_1, H[0 * M:(0 + 1) * M][:, 0 * N:(0 + 1) * N]), P_1[:, 1])) / np.linalg.norm(F_1)) ** 2)
    R_2 =  0.5 * np.log2(1 + (np.squeeze(np.dot(np.dot(F_2, H[1 * M:(1 + 1) * M][:, 1 * N:(1 + 1) * N]), P_2)) / np.linalg.norm(F_2)) ** 2)
    R_3 =  0.5 * np.log2(1 + (np.squeeze(np.dot(np.dot(F_3, H[2 * M:(2 + 1) * M][:, 2 * N:(2 + 1) * N]), P_3)) / np.linalg.norm(F_3)) ** 2)

    return R_1_1 + R_1_2 + R_2 +R_3


if __name__ == "__main__":

    M = 3
    N = 3
    K = 2
    J = 2
    # M = 3
    # N = 6
    # K = 3
    # J = 3
    # SNR = [10 ** 1,10 ** 1.25, 10 ** 1.5, 10 ** 1.75, 10 ** 2, 10 ** 2.25, 10 ** 2.5, 10 ** 2.75, 10 ** 3,10 ** 3.25, 10 ** 3.5]
    # SNR = [10**1, 10**2, 10**2.5, 10**3, 10**3.5,  1e4, 10**4.5, 10**5, 10**5.5, 10**6, 10**6.5,  1e7, 10**7.5, 10**8]
    SNR = [10 ** 0.5, 10 ** 0.75]
    iter = 2000
    p = Pool(20)
    IA_sum_rate_list = []

    for snr in SNR:
        IA_sum_rate = 0
        IA_rate = 0
        multiple_res = []
        for i in range(iter):
            H_gaus = np.random.randn(K * M, K * N)
            # H_gaus = chanMatrix(M, N, K, J)
            # H_gaus = np.eye(K * M)
            res = p.apply_async(rate_IA_two_user, (H_gaus, M, N, snr))
            # res = p.apply_async(rate_IA_three_user, (H_gaus, M, N, snr))
            # IA_rate = rate_IA_two_user(H_gaus, M, N, snr)
            # IA_rate = rate_IA_three_user(H_gaus, M, N, snr)
            # dof_sum_rate = dof_sum_rate + dof_rate
            multiple_res.append(res)
        for res in multiple_res:
            IA_rate = res.get()
            IA_sum_rate = IA_sum_rate + IA_rate
        IA_sum_rate = IA_sum_rate / iter
        IA_sum_rate_list.append(IA_sum_rate)
    Full_Result = np.column_stack((10 * np.log10(SNR), IA_sum_rate_list))
    np.savetxt(
        '/home/haizi/PycharmProjects/SSA/Simu_result/' + 'IA' + ' K=' + K.__str__() + ' iter =' + iter.__str__() + time.ctime() + 'Simu_Data.txt',
        Full_Result, fmt='%1.5e')
    # print 'sum rate of Decode and forward scheme:', dof_sum_rate
    pyplot.plot(10 * np.log10(SNR), IA_sum_rate_list, 'b*-', label='IA')
    pyplot.xlabel('SNR/dB')
    pyplot.ylabel('Sum Rate/bps')
    pyplot.legend(loc='upper left')
    pyplot.savefig('/home/haizi/PycharmProjects/SSA/Simu_result/' +'IA'+ ' K=' + K.__str__() + ' iter =' + iter.__str__() + time.ctime() + 'fig', format = 'eps')
    pyplot.show()