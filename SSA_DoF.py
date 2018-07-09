# !/usr/bin/python2.7
# coding=utf-8
# *************************
# decode and forward scheme for comparison in G-SSA-PNC response
#  identity matrix is used as beamforming matrix in users
#  full decoding is employed in BSs
# *************************
import matplotlib.pyplot as pyplot
from multiprocessing import Pool
import numpy as np
import time
from SSA_fading_channel_model import chanMatrix

def rate_DoF(M, N, K, J, H, SNR):

    sum_rate  = 0
    SINR = np.eye(M)
    for j in range(J):
        H_jj = H[j * M:(j + 1) * M][:, j * N:(j + 1) * N]
        K_list = range(K)
        K_list.remove(j)
        for k in K_list:
            SINR = SINR + (np.sqrt(SNR)/N)**2 * np.dot(H[j * M:(j + 1) * M][:, k * N:(k + 1) * N], np.transpose(H[j * M:(j + 1) * M][:, k * N:(k + 1) * N]))
        sign, logdet =  np.linalg.slogdet(np.ones((M, M)) + np.sqrt( SNR)**2/N**2 * np.dot(np.dot(np.transpose(H_jj), np.linalg.inv(SINR)), H_jj) )
        if sign == 1:
            sum_rate = sum_rate + 0.5 * logdet
        elif sign == 0:
            continue
        elif sign == -1:
            continue
        else:
            raise Exception('Error occurs in logdet computation!!!')

    return max(sum_rate, 0)

if __name__ == "__main__":

    M = 3
    N = 3
    K = 2
    J = 2
    SNR = [10**1, 10**2, 10**2.5, 10**3, 10**3.5,  1e4, 10**4.5, 10**5]
    iter = 2000
    p = Pool(20)
    dof_sum_rate_list = []

    for snr in SNR:
        dof_sum_rate = 0
        dof_rate = 0
        multiple_res = []
        for i in range(iter):
            # H_gaus = np.random.randn(K * M, K * N)
            H_gaus = chanMatrix(M, N, K, J)
            # H_gaus = np.eye(K * M)
            res = p.apply_async(rate_DoF, (M, N, K, J, H_gaus, snr))
            # dof_rate = rate_DoF(M, N, K, J, H_gaus, snr)
            # dof_sum_rate = dof_sum_rate + dof_rate
            multiple_res.append(res)
        for res in multiple_res:
            dof_rate = res.get()
            dof_sum_rate = dof_sum_rate + dof_rate
        dof_sum_rate = dof_sum_rate / iter
        dof_sum_rate_list.append(dof_sum_rate)
    Full_Result = np.column_stack((10 * np.log10(SNR), dof_sum_rate_list))
    np.savetxt(
        '/home/haizi/PycharmProjects/SSA/Simu_result/' + 'DoF' + ' K=' + K.__str__() + ' iter =' + iter.__str__() + time.ctime() + 'Simu_Data.txt',
        Full_Result, fmt='%1.5e')
    # print 'sum rate of Decode and forward scheme:', dof_sum_rate
    pyplot.plot(10 * np.log10(SNR), dof_sum_rate_list, 'b*-', label='DoF')
    pyplot.xlabel('SNR/dB')
    pyplot.ylabel('Sum Rate/bps')
    pyplot.legend(loc='upper left')
    pyplot.savefig('/home/haizi/PycharmProjects/SSA/Simu_result/' +'DoF'+ ' K=' + K.__str__() + ' iter =' + iter.__str__() + time.ctime() + 'fig', format = 'eps')
    pyplot.show()