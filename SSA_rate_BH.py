import os, time
import matplotlib.pyplot as pyplot
from multiprocessing import Pool
from SSA_paramter import *
from SSA_main import *


if __name__ == '__main__':

    SNR = 10 ** 3.0
    # BH_list = [2, 4, 6, 8, 10, 12, 14, 16]
    BH_list = [16]
    Rate_opt_list = [0] * len(BH_list)
    iter = 200
    parallel = Pool(20)

    for i in range(len(BH_list)):
        Capacity_BH = BH_list[i]
        multiple_res = []
        rate0 = 0
        t1 = time.time()
        for ii in range(iter):
            # set random seed
            np.random.seed()
            # random channel matrix
            H = np.random.randn(K * M, K * N)
            # rate optimization for 2*2 system
            res = parallel.apply_async(generation_matrix_optimize, (np.array(list(G_2Full) + list(G_2Full_2)), H, SNR))
            multiple_res.append(res)
        t2 = time.time()
        print "Total time cost:", (t2 - t1), 's.'
        for res in multiple_res:
            [ sum_rate_opt, sum_rate_init, temp] = res.get()
            rate0 = rate0 + sum_rate_opt
        Rate_opt_list[i] = min(rate0 / iter, Capacity_BH * K)
    print 'Sum rate list:', Rate_opt_list
    Full_Result = np.column_stack((BH_list,  Rate_opt_list))
    np.savetxt('/home/haizi/PycharmProjects/SSA/Simu_result/' + 'SSA_Rate-vs-BH' + ' K=' + K.__str__() + ' SNR ' + SNR.__str__() + ' iter =' + iter.__str__() + time.ctime() + 'Simu_Data.txt',
        Full_Result, fmt='%1.5e')
    pyplot.plot(BH_list, Rate_opt_list, 'go-', label='Opt G&DE P_beta')
    pyplot.xlabel('Backhaul Capacity per BS/bit per channel use')
    pyplot.ylabel('Sum Rate/bit per channel use')
    pyplot.legend(loc='upper left')
    pyplot.show()