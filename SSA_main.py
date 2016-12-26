import cvxpy as cvx
import numpy as np
import os, time
import matplotlib.pyplot as pyplot
from multiprocessing import Pool
from scipy import optimize
from SSA_paramter import *





#--------------------------------#
# The achievable rate upbound of one stream of user i, R_{i,l}
# carefully pass parameter into the function
# if index_eta(i,l,j) == None, the constraint don't exist.!!!
#--------------------------------#
def Rate_i_l_Bound(H, P, f, SNR,i,l,j):

    # compute the corresponding channel matrix
    H_j_i = H[j*M:(j+1)*M][:,i*N:(i+1)*N]
    # the column vector p
    p = np.array(P[index_I(i,l)*N:(index_I(i,l)+1)*N])
    #print np.dot(H_j_i, np.expand_dims(p,axis=1))
    # #print 'dimension of multiplication:',(f * np.dot(H_j_i, np.expand_dims(p, axis=1)))
    # if type == 'cvx':
    #     R_i_j = cvx.log(cvx.norm( f * np.dot(H_j_i, np.expand_dims(p, axis=1))) * SNR ) * np.log2(q)
    # elif type == 'array':
    #print np.dot(f, np.dot(H_j_i, np.expand_dims(p, axis=1)))
    R_i_j = max(0, np.log2(np.linalg.norm(np.dot(f, np.dot(H_j_i, np.expand_dims(p, axis=1))))**2 * SNR)) * np.log2(q)
        #R_i_j = max(R_i_j,0)
    # else:
    #     raise Exception('No such type!')
    return R_i_j

# compute the sum rate when given precoding matrix P_init
def sum_rate_fun(H, P_init, SNR):
    # compute the constraint matrix for f_{j,l}
    F_constr_matrix = F_full_constr_matrix(H, P_init)

    # find a feasible non-zero F in the left nullspace of F_constr_matrix
    # 1 represent there are one possible feasible f (we ignore the negtive value here.)
    # (the constraint matrix is 3 * 2 for each f)
    fea_F = np.zeros((1, M * sum(L_node)))
    for ii in range(sum(L_node)):
        [s, v, d] = np.linalg.svd(np.transpose(F_constr_matrix[ii * M:(ii + 1) * M]))
        # print np.expand_dims(d[-1,:]/np.linalg.norm(d[-1,:]), axis=0)
        # print fea_F[:,ii*N:(ii+1)*N]
        fea_F[:, ii * M:(ii + 1) * M] = np.expand_dims(d[-1, :] / np.linalg.norm(d[-1, :]), axis=0)

    # compute the initial rate
    R_init = [0] * sum(L_node)
    for i in range(K):
        for l in range(L_node[i]):
            temp = []
            for j in range(K):
                if index_eta(i, l, j) != None:
                    temp.append(Rate_i_l_Bound(H, P_init,
                                               fea_F[:, index_I(j, index_eta(i, l, j)) * M:(index_I(j, index_eta(i, l, j)) + 1) * M],
                                               SNR, i, l, j))
            R_init[index_I(i, l)] = max(min(temp), 0)
    sum_rate_init = sum(R_init)
    return sum_rate_init

# compute the augmented channel matrix \tilde(H)

'''
# the function is a special case for given G and 6*6 H
def Augmented_chan_matrix(H,G):
    H_aug = np.array([])
    for j in range(K):
        for l in range(L_node[j]):
            supp_ind = index_S(j,l)
            if len(supp_ind) == 2:
                H_N = np.zeros((M, N*sum(L_node)))
                # print H_N[:,supp_ind[0]*N:(supp_ind[0] + 1)*N]
                # print np.array(H[j*N:(j+1)*N][:, index_inv(supp_ind[0])*N:(index_inv(supp_ind[0])+1)*N]) * float(G[index_I(j,l),supp_ind[1]])
                H_N[:,supp_ind[0]*N:(supp_ind[0] + 1)*N] = \
                np.array(H[j*M:(j+1)*M][:, index_inv(supp_ind[0])*N:(index_inv(supp_ind[0])+1)*N]) * float(G[index_I(j,l),supp_ind[1]])
                # print np.array(H[j * N:(j + 1) * N][:, index_inv(supp_ind[1]) * N:(index_inv(supp_ind[1]) + 1) * N]) * float(G[index_I(j, l), supp_ind[0]])
                H_N[:, supp_ind[1] * N:(supp_ind[1] + 1) * N] = \
                np.array(H[j * M:(j + 1) * M][:, index_inv(supp_ind[1]) * N:(index_inv(supp_ind[1]) + 1) * N]) * float(G[index_I(j, l), supp_ind[0]])
                H_aug = np.array(H_aug.tolist() + H_N.tolist())
    return H_aug
'''


def Augmented_chan_matrix(H, G_extra, Beta):
    H_aug = np.array([])
    for j in range(K):
        for l in range(L_node[j]):
            supp_ind = index_S(j,l)
            if len(supp_ind) >= 2:
                H_N = np.zeros((M, N*sum(L_node)))
                # print H_N[:,supp_ind[0]*N:(supp_ind[0] + 1)*N]
                # print np.array(H[j*N:(j+1)*N][:, index_inv(supp_ind[0])*N:(index_inv(supp_ind[0])+1)*N]) * float(G[index_I(j,l),supp_ind[1]])
                for ii in range(len(supp_ind)):
                    if ii == 0:
                        H_N[:,supp_ind[ii]*N:(supp_ind[ii] + 1)*N] = \
                        np.array(H[j*M:(j+1)*M][:, index_inv(supp_ind[ii])*N:(index_inv(supp_ind[ii])+1)*N]) * float(1)
                    # print np.array(H[j * N:(j + 1) * N][:, index_inv(supp_ind[1]) * N:(index_inv(supp_ind[1]) + 1) * N]) * float(G[index_I(j, l), supp_ind[0]])
                    else:
                        H_N[:, supp_ind[ii] * N:(supp_ind[ii] + 1) * N] = \
                        np.array(H[j * M:(j + 1) * M][:, index_inv(supp_ind[ii]) * N:(index_inv(supp_ind[ii]) + 1) * N]) * float(Beta.pop())
                H_aug = np.array(H_aug.tolist() + H_N.tolist())
    # consider the extra serveral aligned "bin signal", specified by G_extra
    m = G_extra.shape[0]
    for jj in range(m):
        S = []
        g = G_extra[jj]
        for ii in range(len(g)):
            if g[ii] != 0:
                S.append(ii)
        supp_ind = S
        if len(supp_ind) >= 2:
            H_N = np.zeros((M, N * sum(L_node)))
            for ii in range(len(supp_ind)):
                if ii == 0:
                    H_N[:, supp_ind[ii] * N:(supp_ind[ii] + 1) * N] = \
                        np.array(H[j * M:(j + 1) * M][:,
                                 index_inv(supp_ind[ii]) * N:(index_inv(supp_ind[ii]) + 1) * N]) * float(1)
                else:
                    H_N[:, supp_ind[ii] * N:(supp_ind[ii] + 1) * N] = \
                        np.array(H[j * M:(j + 1) * M][:,
                                 index_inv(supp_ind[ii]) * N:(index_inv(supp_ind[ii]) + 1) * N]) * float(Beta.pop())
            H_aug = np.array(H_aug.tolist() + H_N.tolist())
    return H_aug

# compute the matrix that constraint receive shaping vector f_{j,l}
def F_constr_mat(H,P,j,l):
    if len(P.shape) == 2:
        P = np.squeeze(P)
    eta = index_S(j,l)[0]
    #H_p = np.zeros((1,N))
    H_p = np.array([])
    for ii in range(L_node[0]):
        if ii != eta:
            #print H[j * N:(j + 1) * N][:, ii * N : (ii + 1) * N]
            #print P[ii*N:(ii+1)*N]
            # print H_p
            # print H[j * N:(j + 1) * N][:, index_I(0,ii)*N:(index_I(0,ii)+1)*N]
            # print P[index_I(0,ii)*N:(index_I(0,ii)+1)*N]
            # print np.expand_dims(np.transpose(np.dot(H[j * N:(j + 1) * N][:, ii * N : (ii + 1) * N], P[index_I(0,ii)*N:(index_I(0,ii)+1)*N])),axis=0).tolist()
            H_p = np.array(H_p.tolist() + np.expand_dims(np.transpose(np.dot(H[j * M:(j + 1) * M][:, 0 : 1 * N], P[index_I(0,ii)*N:(index_I(0,ii)+1)*N])),axis=0).tolist())

    return np.transpose(H_p)

# compute a full constraint matrix for all f_{j,l}
def F_full_constr_matrix(H,P):
    # compute the constrian matrix for each receiving shaping vector
    F_constr_matrix = np.zeros((N*sum(L_node), 2))
    # print F_constr_mat(H, P, 0, 1)
    # print F_constr_matrix[0:3, :]
    F_constr_matrix[0:3, :] = F_constr_mat(H, P, 0, 0)
    F_constr_matrix[3:6, :] = F_constr_mat(H, P, 0, 1)
    F_constr_matrix[6:9, :] = F_constr_mat(H, P, 0, 2)
    F_constr_matrix[9:12, :] = F_constr_mat(H, P, 1, 0)
    F_constr_matrix[12:15, :] = F_constr_mat(H, P, 1, 1)
    if K == 3:
        F_constr_matrix[15:18, :] = F_constr_mat(H, P, 2, 0)
        F_constr_matrix[18:21, :] = F_constr_mat(H, P, 2, 1)
    return F_constr_matrix

def Power_optimize(H,G,SNR):
    # Initialization the alternatively algorithm
    H_tilde = Augmented_chan_matrix(H, G)
    # find a feasible non-zero P in the nullspace of H_tilde
    [s, v, d] = np.linalg.svd(H_tilde)
    fea_P = d[-1,:]
    # normalize P_init
    # P_init = E_total * fea_P / np.linalg.norm(fea_P, 2)
    P_init = fea_P / np.linalg.norm(fea_P, 2)
    print 'P_init:\n', P_init, np.linalg.norm(P_init, 2)

    #Sum_rate_init = sum_rate_fun(H, P_init, SNR)

    # compute the constraint matrix for f_{j,l}
    F_constr_matrix = F_full_constr_matrix(H, P_init)

    # find a feasible non-zero F in the left nullspace of F_constr_matrix
    # 1 represent there are one possible feasible f (we ignore the negtive value here.)
    # (the constraint matrix is 3 * 2 for each f)
    fea_F = np.zeros((1, N * sum(L_node)))
    for ii in range(sum(L_node)):
        [s, v, d] = np.linalg.svd(np.transpose(F_constr_matrix[ii * N:(ii + 1) * N]))
        # print np.expand_dims(d[-1,:]/np.linalg.norm(d[-1,:]), axis=0)
        # print fea_F[:,ii*N:(ii+1)*N]
        fea_F[:, ii * N:(ii + 1) * N] = np.expand_dims(d[-1, :] / np.linalg.norm(d[-1, :]), axis=0)

    # compute the initial rate
    R_init = [0] * sum(L_node)
    for i in range(K):
        for l in range(L_node[i]):
            temp = []
            for j in range(K):
                if index_eta(i, l, j) != None:
                    temp.append(Rate_i_l_Bound(H, P_init, fea_F[:, index_I(j, index_eta(i, l, j)) * N:(index_I(j,index_eta(i,l,j)) + 1) * N],
                                               SNR, i, l, j))
            R_init[index_I(i, l)] = max(min(temp), 0)
    Sum_rate_init = sum(R_init)

    if True:
        # compute the orthogonal basis for nullspace
        dimen_Null = d.shape[0] - len(v)
        basis_Null = d[len(v):d.shape[0], :]
        [orth_basis, k] = np.linalg.qr(np.transpose(basis_Null))

        # represent P as a combination of orth_basis
        P_coef = cvx.Variable(dimen_Null)
        P = np.zeros((orth_basis.shape[0],1))
        for ii in range(dimen_Null):
            P = P + P_coef[ii] * orth_basis[:,ii]

    # normalize the precoding vector and then optimize the power
    P_unit = np.zeros((len(P_init)))
    for ii in range(sum(L_node)):
        P_unit[ii*N:(ii+1)*N] = P_init[ii*N:(ii+1)*N]/np.linalg.norm(P_init[ii*N:(ii+1)*N])

    # the power of precoding vector
    p_pow = cvx.Variable(sum(L_node))
    Constr = [cvx.sum_entries(p_pow) <= 1]

    # objective function
    Obj = 0
    Obj_init = 0
    # compute the objective function
    len_f_H = []
    F_H = np.zeros((1, 3))
    for i in range(K):
        for l in range(L_node[i]):
            f_H = np.zeros((1, 3))
            for j in range(K):
                l_prime = index_eta(i, l, j)
                if l_prime != None:
                    f_H = np.vstack([f_H, np.dot(fea_F[:, index_I(j, l_prime) * N:(index_I(j, l_prime) + 1) * N],
                                                 H[j * N:(j + 1) * N][:, i * N:(i + 1) * N])])
            f_H = np.delete(f_H, 0, 0)
            len_f_H.append(f_H.shape[0])
            F_H = np.concatenate((F_H, f_H), axis=0)
            #Obj = Obj + cvx.log(cvx.min_entries(cvx.norm(f_H * P_unit[index_I(i, l) * N: (index_I(i, l) + 1) * N],2) * p_pow[index_I(i,j)]))
            #print index_I(i,l)
            Obj = Obj + cvx.log(p_pow[index_I(i,l)])
            Constr.append(p_pow[index_I(i,l)] >= 1/min(np.power(np.dot(f_H, P_unit[index_I(i, l) * N: (index_I(i, l) + 1) * N]),2)*SNR))
            #Obj_init = Obj_init + min(np.power(np.dot(f_H, P_init[index_I(i, l) * N: (index_I(i, l) + 1) * N]),2))
            Obj_init = Obj_init + np.log2(np.linalg.norm(P_init[index_I(i, l) * N: (index_I(i, l) + 1) * N])**2)
    F_H = np.delete(F_H, 0, 0)

    Obj_func = cvx.Maximize(Obj)
    Prob = cvx.Problem(Obj_func, Constr)
    Prob.solve( verbose=True)
    P_pow_value = p_pow.value

    if Prob.status != 'optimal':
        return 0,0
    print 'Problme status:', Prob.status
    print 'vector power:', P_pow_value, np.linalg.norm(P_pow_value,1)
    print 'optimal objective value', Prob.value
    print 'inital objective value:', Obj_init


    # compute the true rate
    R_opt = 0
    Sum_R_init = 0
    for ii in range(len(len_f_H)):
        # print F_H[sum(len_f_H[0:ii]):sum(len_f_H[0:ii+1])]
        # print P_init[ii * N: (ii + 1) * N]
        R_i_l = np.log2(min(np.power(
            np.dot(F_H[sum(len_f_H[0:ii]):sum(len_f_H[0:ii + 1])], P_unit[ii * N: (ii + 1) * N]), 2)) * P_pow_value[ii] * SNR) * np.log2(q)
        R_i_l_init = np.log2(min(np.power(
            np.dot(F_H[sum(len_f_H[0:ii]):sum(len_f_H[0:ii + 1])], P_init[ii * N: (ii + 1) * N]), 2)) * SNR) * np.log2(q)
        print R_i_l, R_i_l_init
        R_opt = R_opt + max(R_i_l, 0)
        Sum_R_init = Sum_R_init + max(R_i_l_init, 0)
    if type(R_opt) != int:
        R_opt = R_opt.tolist()[0]

    print 'final computed rate:', np.squeeze(R_opt), Sum_rate_init

    return np.squeeze(R_opt), Sum_rate_init


def Bilinear_optimize(H,G):

    # Initialization the alternatively algorithm
    H_tilde = Augmented_chan_matrix(H, G)
    # find a feasible non-zero P in the nullspace of H_tilde
    [s, v, d] = np.linalg.svd(H_tilde)
    fea_P = d[-1, :]
    P_init = E_total * fea_P / np.linalg.norm(fea_P, 2)
    print 'P_init:\n', P_init
    print 'H_tilde * P_init:\n', np.dot(H_tilde,P_init)

    # compute the constraint matrix for f_{j,l}
    F_constr_matrix = F_full_constr_matrix(H,P_init)

    # find a feasible non-zero F in the left nullspace of F_constr_matrix
    # 1 represent there are one possible feasible f (we ignore the negtive value here.)
    # (the constraint matrix is 3 * 2 for each f)
    fea_F = np.zeros((1,N*sum(L_node)))
    for ii in range(sum(L_node)):
        [s,v,d] = np.linalg.svd(np.transpose(F_constr_matrix[ii*N:(ii+1)*N]))
        #print np.expand_dims(d[-1,:]/np.linalg.norm(d[-1,:]), axis=0)
        #print fea_F[:,ii*N:(ii+1)*N]
        fea_F[:,ii*N:(ii+1)*N] = np.expand_dims(d[-1,:]/np.linalg.norm(d[-1,:]), axis=0)

    print 'fea_F * F_constr_matrix:\n', np.dot(fea_F,F_constr_matrix)
    # compute the initial rate
    R_init = [0]*sum(L_node)

    for i in range(K):
        for l in range(L_node[i]):
            temp = []
            for j in range(K):
                if index_eta(i, l, j) != None:
                    temp.append( Rate_i_l_Bound(H, P_init, fea_F[:,index_I(j, index_eta(i, l, j))*N:(index_I(j, index_eta(i, l, j)) + 1)*N], 'array', i, l, j))
            R_init[index_I(i, l)] = max(min(temp),0)

    Sum_rate_init = sum(R_init)


    # apply cvxpy to solve the relaxed optimization problem
    # Define the variables
    P = cvx.Variable(N * sum(L_node),1)

    # compute all constraints
    Constr = [H_tilde * P == 0, cvx.norm(P,2) <= E_total]
    if (np.linalg.norm(np.dot(H_tilde,P_init)) >= 1e-8) | (np.linalg.norm(P_init) >= (1 + 1e-3) * E_total):
        raise Exception('The NullSpace constraint is Wrong!!!')
    # compute the index set of f that constriant p
    if True:
        #index_P = [0]*L_node[0]
        for ii in range(L_node[0]):
            temp = []
            for jj in range(sum(L_node)):
                #print G[jj,ii]
                if G[jj,ii] == 0:
                    temp.append(jj)
                    # the constraints for P

                    Constr.append( np.dot(fea_F[:,jj*N:(jj+1)*N], H[index_inv(jj)*N:(index_inv(jj)+1)*N][:,0:N]) * P[ii*N:(ii+1)*N] == 0)
                    # check the constraints
                    if np.linalg.norm(np.dot(np.dot(fea_F[:, jj * N:(jj + 1) * N], H[index_inv(jj) * N:(index_inv(jj) + 1) * N][:, 0:N]), P_init[ii*N:(ii+1)*N])) >= 1e-5:
                        raise  Exception('The NullSpace constraint is Wrong!!!')
                    # tt = np.ones((1,3))
                    # Constr.append(tt * P[ii * N:(ii + 1) * N] == 0)
            #index_P[ii] = temp

    #objective function
    Obj = 0
    Obj_temp = 0
    Obj_init = 0
    temp =cvx.Variable(sum(L_node),1)
    # compute the objective function
    len_f_H = []
    F_H = np.zeros((1,3))
    for i in range(K):
        for l in range(L_node[i]):
            f_H = np.zeros((1,3))
            for j in range(K):
                l_prime = index_eta(i, l, j)
                if l_prime != None:
                    #print fea_F[:, index_I(j, l_prime) * N:(index_I(j, l_prime) + 1) * N]
                    #print H[j * N:(j + 1) * N][:, i * N:(i + 1) * N]
                    #print np.dot(fea_F[:, index_I(j, l_prime) * N:(index_I(j, l_prime) + 1) * N], H[j * N:(j + 1) * N][:, i * N:(i + 1) * N])
                    f_H = np.vstack([f_H, np.dot(fea_F[:,index_I(j,l_prime)*N:(index_I(j,l_prime) + 1)*N],H[j*N:(j+1)*N][:,i*N:(i+1)*N])])
                    #print f_H
            f_H = np.delete(f_H,0,0)
            len_f_H.append(f_H.shape[0])
            Constr.extend([f_H * P[index_I(i,l)*N : (index_I(i,l) + 1)*N] >= np.ones((f_H.shape[0],1)) * temp[index_I(i,l)]])
            F_H = np.concatenate((F_H,f_H),axis=0)
            Obj = Obj + cvx.min_entries(f_H * P[index_I(i,l)*N : (index_I(i,l) + 1)*N])
            Obj_init = Obj_init + min(np.abs(np.dot(f_H, P_init[index_I(i,l)*N : (index_I(i,l) + 1)*N])))

    F_H = np.delete(F_H,0,0)
    Obj_temp = cvx.sum_entries(temp)
    # solve the maximization problem
    #Object = cvx.Maximize(Obj)
    Object = cvx.Maximize(Obj_temp)
    Prob = cvx.Problem(Object, Constr)
    Prob.solve(solver = cvx.MOSEK, verbose=True)

    print 'Problme status:',Prob.status
    print 'optimal objective value',Prob.value
    print 'inital objective value:',Obj_init
    P_opt = (P.value).A
    #P_opt = E_total * P_opt / np.linalg.norm(P_opt, 2)

    # check the constraints
    print 'Norm(P_opt - P_init):\n', np.linalg.norm(P_opt) - np.linalg.norm(P_init)
    print 'H_tild * P_opt:\n', np.dot(H_tilde,P_opt)
    print 'the optimal precoding vector:\n',np.squeeze(P_opt)
    print 'the l_2 norm of P:\n', np.linalg.norm(P_opt,2)
    print 'fea_F * F_constr_matrix:\n', np.dot(fea_F,F_full_constr_matrix(H,np.squeeze(P_opt)))


    # compute the true sum rate using Rate_i_l_Bound() function
    R_opt = [0] * sum(L_node)

    for i in range(K):
        for l in range(L_node[i]):
            temp = []
            for j in range(K):
                if index_eta(i, l, j) != None:
                    temp.append(Rate_i_l_Bound(H, np.squeeze(P_opt), fea_F[:, index_I(j, index_eta(i, l, j)) * N:(index_I(j, index_eta(i, l,j)) + 1) * N], 'array', i, l, j))
            R_opt[index_I(i, l)] = max(min(temp), 0)

    Sum_R_opt = sum(R_opt)

    # compute the true sum rate using the objective function
    if True:
        R_opt = 0
        Sum_R_init = 0
        for ii in range(len(len_f_H)):
            #print F_H[sum(len_f_H[0:ii]):sum(len_f_H[0:ii+1])]
            # print P_init[ii * N: (ii + 1) * N]
            R_i_l = np.log2(min(np.power(
                    np.dot(F_H[sum(len_f_H[0:ii]):sum(len_f_H[0:ii+1])], P_opt[ii * N: (ii + 1) * N]),2))* SNR) * np.log2(q)
            R_i_l_init = np.log2(min(np.power(
                    np.dot(F_H[sum(len_f_H[0:ii]):sum(len_f_H[0:ii+1])], P_init[ii * N: (ii + 1) * N]),2)) * SNR) * np.log2(q)
            #print R_i_l
            R_opt = R_opt + max(R_i_l, 0)
            Sum_R_init = Sum_R_init + max(R_i_l_init,0)
        if type(R_opt) != int:
            R_opt = R_opt.tolist()[0]

    return Sum_R_opt,  Sum_rate_init , R_opt, Sum_R_init


def Precoding_Direction_optimize(H,Beta, SNR, iter):

    # when K =3, there are loops that we need consider
    Beta = list(Beta)
    if K == 3:
        #Beta = list(np.random.rand(10))
        '''
        Beta.insert(5, 1)
        Beta.insert(6, 1)
        # compute the two constrainted beta,  i.e., beta_6 and beta_7
        Beta[5] = Beta[1] * Beta[3] * G[5][1] / (Beta[0] * G[5][5])
        Beta[6] = Beta[3] * Beta[4] * Beta[7] * G[4][6] / (Beta[0]  * Beta[2] * G[4][0])
        '''
        Beta.insert(1, 1)
        Beta.insert(2, 1)
        Beta.insert(6, 1)
        # compute the three constrainted beta,  i.e., beta_2, beta_3, and beta_7
        Beta[1] = Beta[8] * G[5][0] / G[5][5]
        Beta[2] = Beta[5] * G[1][4] / G[1][1]
        Beta[6] = Beta[5] * Beta[8] * G[4][4] / (Beta[7] * G[4][5])

    # Initialization the alternatively algorithm
    H_tilde = Augmented_chan_matrix(H, G_extra, Beta)
    # find the nullspace of H_tilde
    [s, v, d] = np.linalg.svd(H_tilde)

    # compute the orthogonal basis for nullspace
    dimen_Null = d.shape[0] - len(v)
    basis_Null = d[len(v):d.shape[0], :]
    orth_basis = np.transpose(np.array(basis_Null))
    #[orth_basis, k] = np.linalg.qr(np.transpose(basis_Null))

    '''
    # find a P to compare
    sum_rate_init = 0
    P_coef = np.random.rand(dimen_Null)
    fea_P = np.dot(orth_basis, np.transpose(P_coef))
    #fea_P = d[-1, :]
    P_init = fea_P / np.linalg.norm(fea_P, 2)
    # compute the sum rate when P = P_init
    sum_rate_init = sum_rate_fun(H, P_init, SNR)
    print 'init rate:', sum_rate_init
    '''

    # represent P as a combination of orth_basis
    sum_rate_opt = 0
    max_sum_rate_random = 0
    max_rate_P = 0
    np.random.seed()
    for ii in range(iter):
        P_coef = np.random.rand(dimen_Null)
        # P = np.zeros((orth_basis.shape[0], 1))
        # P_random = np.dot(orth_basis, np.transpose(P_coef))
        # P_random = P_random / np.linalg.norm(P_random, 2)

        P_coef = np.random.rand(dimen_Null, 1)
        P_random = np.squeeze(np.dot(orth_basis, P_coef))
        # fea_P = d[-1, :]
        P_random_norm = np.linalg.norm(P_random, 2)
        P_random = P_random / np.linalg.norm(P_random, 2)
        P_random_norm = np.linalg.norm(P_random, 2)
        #print 'l2-norm of H_tilde * P_random: ', np.linalg.norm(np.dot(H_tilde,P_random))
        # compute the sum rate when P = P_random
        sum_rate_opt = sum_rate_fun(H, P_random, SNR)
        #print '\n random rate:', sum_rate_opt
        if sum_rate_opt > max_sum_rate_random:
            max_rate_P = P_random
            max_sum_rate_random = sum_rate_opt

    sum_rate_opt = max_sum_rate_random

    return sum_rate_opt

'''
Function: compute the sum rate with given precoding vector
Input: basis for feasible precoding vector P and coefficients of P
Output: sum rate
'''

def precoding_vector_coefficient(P_coef,Orth_basis,H,SNR):

    # compute the combination
    P_random = np.dot(Orth_basis, np.transpose(P_coef))
    P_random = P_random / np.linalg.norm(P_random, 2)
    Sum_rate_random = sum_rate_fun(H, P_random, SNR)
    return Sum_rate_random # to support the differential evolution funciton

'''
Function: optimize precoding vector with differential evolution (scipy)
Input   : channel matrix H, and SNR
Output  : optimal (maximum) sum rate
'''

def precoding_vector_optimize_DE(H, Beta_init, SNR):
    Beta_init = list(Beta_init)
    if K == 3:
        # Beta = list(np.random.rand(10))
        '''
        Beta_init.insert(5, 1)
        Beta_init.insert(6, 1)
        # compute the two constrainted beta,  i.e., beta_6 and beta_7
        Beta_init[5] = Beta_init[1] * Beta_init[3] * G[5][1]/ (Beta_init[0] * G[5][5])
        Beta_init[6] = Beta_init[3] * Beta_init[4] * Beta_init[7] * G[4][6]/ (Beta_init[0] * Beta_init[2] * G[4][0])
        '''
        Beta_init.insert(1, 1)
        Beta_init.insert(2, 1)
        Beta_init.insert(6, 1)
        # compute the three constrainted beta,  i.e., beta_2, beta_3, and beta_7
        Beta_init[1] = Beta_init[8] * G[5][0] / G[5][5]
        Beta_init[2] = Beta_init[5] * G[1][4] / G[1][1]
        Beta_init[6] = Beta_init[5] * Beta_init[8] * G[4][4] / (Beta_init[7] * G[4][5])

    H_tilde = Augmented_chan_matrix(H, G_extra, Beta_init)
    # find the nullspace of H_tilde
    [s, v, d] = np.linalg.svd(H_tilde)
    # compute the orthogonal basis for nullspace
    dimen_Null = d.shape[0] - len(v)
    basis_Null = d[len(v):d.shape[0], :]
    basis_Null = np.transpose(np.array(basis_Null))

    # optimize precodeing vector with differential evolution algorithm and identity Beta
    coef_DE_port = lambda x: -precoding_vector_coefficient(x[0:dimen_Null], basis_Null, H, SNR)
    pranges = ((0, 10),) * dimen_Null
    Result_DE = optimize.differential_evolution(coef_DE_port, pranges, maxiter=20, disp=False)
    print 'Precodeing Vector Differential Evolution Status:', Result_DE.success
    sum_rate_opt_vec = -Result_DE.fun

    return sum_rate_opt_vec


'''
Function: optimize precoding vector and beta with differential evolution (scipy)
Input   : channel matrix H, and SNR
Output  : optimal (maximum) sum rate
'''

def precoding_vector_beta_optimize_DE(H, Beta_init, vect_coeff, SNR):
    Beta_init = list(Beta_init)
    if K == 3:
        # Beta = list(np.random.rand(10))
        '''
        Beta_init.insert(5, 1)
        Beta_init.insert(6, 1)
        # compute the two constrainted beta,  i.e., beta_6 and beta_7
        Beta_init[5] = Beta_init[1] * Beta_init[3] * G[5][1]/ (Beta_init[0] * G[5][5])
        Beta_init[6] = Beta_init[3] * Beta_init[4] * Beta_init[7] * G[4][6]/ (Beta_init[0] * Beta_init[2] * G[4][0])
        '''
        Beta_init.insert(1, 1)
        Beta_init.insert(2, 1)
        Beta_init.insert(6, 1)
        # compute the three constrainted beta,  i.e., beta_2, beta_3, and beta_7
        Beta_init[1] = Beta_init[8] * G[5][0] / G[5][5]
        Beta_init[2] = Beta_init[5] * G[1][4] / G[1][1]
        Beta_init[6] = Beta_init[5] * Beta_init[8] * G[4][4] / (Beta_init[7] * G[4][5])

    H_tilde = Augmented_chan_matrix(H, G_extra, Beta_init)
    # find the nullspace of H_tilde
    [s, v, d] = np.linalg.svd(H_tilde)
    # compute the orthogonal basis for nullspace
    dimen_Null = d.shape[0] - len(v)
    basis_Null = d[len(v):d.shape[0], :]
    basis_Null = np.transpose(np.array(basis_Null))

    sum_rate_opt_vec_beta = precoding_vector_coefficient(vect_coeff, basis_Null, H, SNR)
    return sum_rate_opt_vec_beta

'''
Function: optimize beta (differential evolution) and precoding vector (naive greedy, choose the best of iter (10) kinds)
Input   :
Output  :
'''

def sum_rate_optimize_beta_precoding(H, SNR):

    # the number of parameter beta
    beta_free = 0
    beta_fix = 0
    if K == 3:
        beta_free = 10 # 12 - 2(numbers of loop)
        beta_fix = 2
    elif K == 2:
        beta_free = 4

    # compute the initial sum rate with random precoding vector and identity Beta
    Beta_init = [1] * (beta_free )
    if K == 3:
        # Beta = list(np.random.rand(10))
        '''
        Beta_init.insert(5, 1)
        Beta_init.insert(6, 1)
        # compute the two constrainted beta,  i.e., beta_6 and beta_7
        Beta_init[5] = Beta_init[1] * Beta_init[3] * G[5][1]/ (Beta_init[0] * G[5][5])
        Beta_init[6] = Beta_init[3] * Beta_init[4] * Beta_init[7] * G[4][6]/ (Beta_init[0] * Beta_init[2] * G[4][0])
        '''
        Beta_init.insert(1, 1)
        Beta_init.insert(2, 1)
        Beta_init.insert(6, 1)
        # compute the three constrainted beta,  i.e., beta_2, beta_3, and beta_7
        Beta_init[1] = Beta_init[8] * G[5][0] / G[5][5]
        Beta_init[2] = Beta_init[5] * G[1][4] / G[1][1]
        Beta_init[6] = Beta_init[5]  * Beta_init[8] * G[4][4] / (Beta_init[7] * G[4][5])

    H_tilde = Augmented_chan_matrix(H, G_extra, Beta_init)
    # find the nullspace of H_tilde
    [s, v, d] = np.linalg.svd(H_tilde)
    # compute the orthogonal basis for nullspace
    dimen_Null = d.shape[0] - len(v)
    basis_Null = d[len(v):d.shape[0], :]
    basis_Null = np.transpose(np.array(basis_Null))
    # generate a P randomly as the comparison object
    P_coef = np.random.rand(dimen_Null,1)
    fea_P = np.squeeze(np.dot(basis_Null, P_coef))
    # fea_P = d[-1, :]
    P_init = fea_P / np.linalg.norm(fea_P, 2)
    # compute the sum rate when P = P_init
    sum_rate_init = sum_rate_fun(H, P_init, SNR)
    '''
    # optimize precodeing vector with differential evolution algorithm and identity Beta
    coef_DE_port = lambda x: precoding_vector_coefficient(x[0:dimen_Null], basis_Null, H, SNR)
    pranges = ((0, 10),) * dimen_Null
    Result_DE = optimize.differential_evolution(coef_DE_port, pranges, maxiter=50, disp=False)
    print 'Precodeing Vector Differential Evolution Status:', Result_DE.success
    sum_rate_opt_vec = -Result_DE.fun
    '''
    # optimize Beta with differential evolution algorithm
    beta_test = [1] * beta_free
    t0 = time.time()
    sum_rate_opt_vec = precoding_vector_optimize_DE(H, beta_test, SNR)
    t1 = time.time()
    print 'Time cost of coefficient optimization:', (t1 - t0)
    #sum_rate_opt_beta_vec_test = Precoding_Direction_optimize(H, [1] * beta_free, SNR, 500)
    diff_evolu_func = lambda beta_coeff: -precoding_vector_beta_optimize_DE(H, beta_coeff[0:beta_free], beta_coeff[beta_free:beta_free+dimen_Null], SNR)
    beta_ranges = ((0.1, 5.0),) * beta_free + ((0,10),) * dimen_Null
    diff_evolu_res = optimize.differential_evolution(diff_evolu_func, beta_ranges, maxiter = 50, disp= False)
    t2 = time.time()
    print 'Beta & Vect_coeff Differential Evolution Status:', diff_evolu_res.success
    sum_rate_opt_beta_vec = -diff_evolu_res.fun
    print 'Time cost of beta&coefficient optimization', (t2 - t1)
    return sum_rate_opt_beta_vec, sum_rate_opt_vec, sum_rate_init

#
# def precoding_vector_optimize_DE(H, Beta_init, SNR):
#     Beta_init = list(Beta_init)
#     if K == 3:
#         # Beta = list(np.random.rand(10))
#         '''
#         Beta_init.insert(5, 1)
#         Beta_init.insert(6, 1)
#         # compute the two constrainted beta,  i.e., beta_6 and beta_7
#         Beta_init[5] = Beta_init[1] * Beta_init[3] * G[5][1]/ (Beta_init[0] * G[5][5])
#         Beta_init[6] = Beta_init[3] * Beta_init[4] * Beta_init[7] * G[4][6]/ (Beta_init[0] * Beta_init[2] * G[4][0])
#         '''
#         Beta_init.insert(1, 1)
#         Beta_init.insert(2, 1)
#         Beta_init.insert(6, 1)
#         # compute the three constrainted beta,  i.e., beta_2, beta_3, and beta_7
#         Beta_init[1] = Beta_init[8] * G[5][0] / G[5][5]
#         Beta_init[2] = Beta_init[5] * G[1][4] / G[1][1]
#         Beta_init[6] = Beta_init[5] * Beta_init[8] * G[4][4] / (Beta_init[7] * G[4][5])
#
#     H_tilde = Augmented_chan_matrix(H, G_extra, Beta_init)
#     # find the nullspace of H_tilde
#     [s, v, d] = np.linalg.svd(H_tilde)
#     # compute the orthogonal basis for nullspace
#     dimen_Null = d.shape[0] - len(v)
#     basis_Null = d[len(v):d.shape[0], :]
#     basis_Null = np.transpose(np.array(basis_Null))
#
#     # optimize precodeing vector with differential evolution algorithm and identity Beta
#     coef_DE_port = lambda x: precoding_vector_coefficient(x[0:dimen_Null], basis_Null, H, SNR)
#     pranges = ((0, 10),) * dimen_Null
#     Result_DE = optimize.differential_evolution(coef_DE_port, pranges, maxiter=20, disp=False)
#     print 'Precodeing Vector Differential Evolution Status:', Result_DE.success
#     sum_rate_opt_vec = -Result_DE.fun
#
#
#     return sum_rate_opt_vec
#
#
# '''
# Function: optimize beta (differential evolution) and precoding vector (naive greedy, choose the best of iter (10) kinds)
# Input   :
# Output  :
# '''
#
# def sum_rate_optimize_beta_precoding(H, SNR):
#
#     # the number of parameter beta
#     beta_free = 0
#     beta_fix = 0
#     if K == 3:
#         beta_free = 10 # 12 - 2(numbers of loop)
#         beta_fix = 2
#     elif K == 2:
#         beta_free = 4
#
#     # compute the initial sum rate with random precoding vector and identity Beta
#     Beta_init = [1] * (beta_free )
#     if K == 3:
#         # Beta = list(np.random.rand(10))
#         '''
#         Beta_init.insert(5, 1)
#         Beta_init.insert(6, 1)
#         # compute the two constrainted beta,  i.e., beta_6 and beta_7
#         Beta_init[5] = Beta_init[1] * Beta_init[3] * G[5][1]/ (Beta_init[0] * G[5][5])
#         Beta_init[6] = Beta_init[3] * Beta_init[4] * Beta_init[7] * G[4][6]/ (Beta_init[0] * Beta_init[2] * G[4][0])
#         '''
#         Beta_init.insert(1, 1)
#         Beta_init.insert(2, 1)
#         Beta_init.insert(6, 1)
#         # compute the three constrainted beta,  i.e., beta_2, beta_3, and beta_7
#         Beta_init[1] = Beta_init[8] * G[5][0] / G[5][5]
#         Beta_init[2] = Beta_init[5] * G[1][4] / G[1][1]
#         Beta_init[6] = Beta_init[5]  * Beta_init[8] * G[4][4] / (Beta_init[7] * G[4][5])
#     H_tilde = Augmented_chan_matrix(H, G_extra, Beta_init)
#     # find the nullspace of H_tilde
#     [s, v, d] = np.linalg.svd(H_tilde)
#     # compute the orthogonal basis for nullspace
#     dimen_Null = d.shape[0] - len(v)
#     basis_Null = d[len(v):d.shape[0], :]
#     basis_Null = np.transpose(np.array(basis_Null))
#     # generate a P randomly as the comparison object
#     P_coef = np.random.rand(dimen_Null,1)
#     fea_P = np.squeeze(np.dot(basis_Null, P_coef))
#     # fea_P = d[-1, :]
#     P_init = fea_P / np.linalg.norm(fea_P, 2)
#     # compute the sum rate when P = P_init
#     sum_rate_init = sum_rate_fun(H, P_init, SNR)
#     '''
#     # optimize precodeing vector with differential evolution algorithm and identity Beta
#     coef_DE_port = lambda x: precoding_vector_coefficient(x[0:dimen_Null], basis_Null, H, SNR)
#     pranges = ((0, 10),) * dimen_Null
#     Result_DE = optimize.differential_evolution(coef_DE_port, pranges, maxiter=50, disp=False)
#     print 'Precodeing Vector Differential Evolution Status:', Result_DE.success
#     sum_rate_opt_vec = -Result_DE.fun
#     '''
#     # optimize Beta with differential evolution algorithm
#     beta_test = [1] * beta_free
#     t0 = time.time()
#     sum_rate_opt_vec = precoding_vector_optimize_DE(H, beta_test, SNR)
#     t1 = time.time()
#     print 'Time cost of coefficient optimization:', (t1 - t0)
#     sum_rate_opt_beta_vec_test = Precoding_Direction_optimize(H, [1] * beta_free, SNR, 500)
#     diff_evolu_func = lambda beta: -precoding_vector_optimize_DE(H, beta, SNR)
#     beta_ranges = ((0.1, 2.0),) * beta_free
#     diff_evolu_res = optimize.differential_evolution(diff_evolu_func, beta_ranges, maxiter = 20, disp= False)
#     t2 = time.time()
#     print 'Beta Differential Evolution Status:', diff_evolu_res.success
#     sum_rate_opt_beta_vec = -diff_evolu_res.fun
#     print 'Time cost of beta&coefficient optimization', (t2 - t1)
#     return sum_rate_opt_beta_vec, sum_rate_opt_vec, sum_rate_init


if __name__ == '__main__':


    #print 'channel matrix:\n', H
    #print Bilinear_optimize(H,G)
#    print Power_optimize(H,G)

    SNR = [1e2, 1e3, 1e4, 1e5, 1e6]
    #SNR = [1e6]
    Rate_pow_opt_list = [0] * len(SNR)
    Rate_opt_list = [0] * len(SNR)
    Rate_init_list = [0] * len(SNR)
    Rate_opt_power_list = [0] * len(SNR)
    iter = 200
    print 'Parent Process %s.' % os.getpid()
    p = Pool(2)

    for i in range(len(SNR)):
        snr = SNR[i]
        rate0 = 0
        rate1 = 0
        rate2 = 0
        multiple_res = []
        # multiple_res = [p.apply_async(precoding_vector_optimize_DE, (H, G, snr)) for i in range(iter)]

        for ii in range(iter):
            # set random seed
            np.random.seed(1)
            # random channel matrix
            H = np.random.randn(K*M, K*N)
            res = sum_rate_optimize_beta_precoding(H, snr)
            #res = p.apply_async(precoding_vector_optimize_DE, (H, snr))
            res = p.apply_async(sum_rate_optimize_beta_precoding, (H, snr))
            multiple_res.append(res)
            # [sum_rate_opt, sum_rate_init] = res.get()
            # [sum_rate_opt, sum_rate_init] = precoding_vector_optimize_DE(H, G, snr)
            # rate0 = rate0 + sum_rate_opt
            # rate1 = rate1 + sum_rate_init
        for res in multiple_res:
            [sum_rate_opt, sum_rate_init, sum_rate_random_pow] = res.get()
            rate0 = rate0 + sum_rate_opt
            rate1 = rate1 + sum_rate_init
            rate2 = rate2 + sum_rate_random_pow
        Rate_opt_list[i]  = rate0/iter
        Rate_init_list[i] = rate1/iter
        Rate_opt_power_list[i] = rate2/iter

    #pyplot.plot(10 * np.log10(SNR), Rate_pow_opt_list, 'rd-', label='Suboptimal P&Pow')

    pyplot.plot(10 * np.log10(SNR), Rate_init_list, 'b*-', label= 'DE P' )
    pyplot.plot(10 * np.log10(SNR), Rate_opt_list, 'go-', label= 'DE Beta & Greedy P')
    pyplot.plot(10 * np.log10(SNR), Rate_opt_power_list, 'kd-', label='Random P')
    '''
    for i in range(len(SNR)):
        snr = SNR[i]
        rate0 = 0
        rate1 = 0
        rate2 = 0
        for ii in range(iter):
            # set random seed
            #np.random.seed()
            # random channel matrix
            H = np.random.randn(6, 6)
            #[R_opt, Sum_rate_init] = Power_optimize(H, G, snr)
            [R_pow_opt, R_opt, Sum_rate_init] = Precoding_Direction_optimize(H, G, snr,5)
            rate0 = rate0 + R_pow_opt
            rate1 = rate1 + R_opt
            rate2 = rate2 + Sum_rate_init
        Rate_pow_opt_list[i] = rate0/iter
        Rate_opt_list[i] = rate1/iter
        rate_init_list[i] = rate2/iter
    #pyplot.plot(10 * np.log10(SNR), Rate_pow_opt_list, 'rd-', label='Suboptimal P&Pow')
    pyplot.plot(10 * np.log10(SNR), Rate_opt_list, 'yo-', label='Suboptimal P-5')

    for i in range(len(SNR)):
        snr = SNR[i]
        rate0 = 0
        rate1 = 0
        rate2 = 0
        for ii in range(iter):
            # set random seed
            #np.random.seed()
            # random channel matrix
            H = np.random.randn(6, 6)
            #[R_opt, Sum_rate_init] = Power_optimize(H, G, snr)
            [R_pow_opt, R_opt, Sum_rate_init] = Precoding_Direction_optimize(H, G, snr,10)
            rate0 = rate0 + R_pow_opt
            rate1 = rate1 + R_opt
            rate2 = rate2 + Sum_rate_init
        Rate_pow_opt_list[i] = rate0/iter
        Rate_opt_list[i] = rate1/iter
        rate_init_list[i] = rate2/iter
    #pyplot.plot(10 * np.log10(SNR), Rate_pow_opt_list, 'rd-', label='Suboptimal P&Pow')
    pyplot.plot(10 * np.log10(SNR), Rate_opt_list, 'ko-', label='Suboptimal P-10')
    for i in range(len(SNR)):
        snr = SNR[i]
        rate0 = 0
        rate1 = 0
        rate2 = 0
        for ii in range(iter):
            # set random seed
            #np.random.seed()
            # random channel matrix
            H = np.random.randn(6, 6)
            #[R_opt, Sum_rate_init] = Power_optimize(H, G, snr)
            [R_pow_opt, R_opt, Sum_rate_init] = Precoding_Direction_optimize(H, G, snr,200)
            rate0 = rate0 + R_pow_opt
            rate1 = rate1 + R_opt
            rate2 = rate2 + Sum_rate_init
        Rate_pow_opt_list[i] = rate0/iter
        Rate_opt_list[i] = rate1/iter
        rate_init_list[i] = rate2/iter
    #pyplot.plot(10 * np.log10(SNR), Rate_pow_opt_list, 'rd-', label='Suboptimal P&Pow')
    pyplot.plot(10 * np.log10(SNR), Rate_opt_list, 'ro-', label='Suboptimal P-100')
    '''
    pyplot.xlabel('SNR/dB')
    pyplot.ylabel('Sum Rate/bps')
    pyplot.legend(loc = 'upper left')
    pyplot.show()