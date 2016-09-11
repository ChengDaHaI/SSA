import numpy as np
import cvxpy as cvx
from SSA_paramter import *



#--------------------------------#
# The achievable rate upbound of one user
# Assume the SNR is very high, then the notation '+' is omitted.
# input: H is the channel matrix from user i to BS j
# input: P is the precoding matrix at user i
# input: F is the total receive shaping matrix
#--------------------------------#
def User_Rate_Bound(H, P_i, F, SNR, i):

    R_i_l = [0] * len(L_node[i])
    min_r_j = 0

    for l in range(L_node[i]):
        for j in range(K):
            # the row vector f
            if None == index_eta(i,l,j):
                break
            else:
                f = np.array(F[index_I(j, index_eta(i,l,j))])
            # the column vector p
            p = np.array(P_i[:,l])
            r_j = -np.log2(np.linalg.norm( np.dot(f, np.dot(H, p))) * SNR ) * np.log2(q)
            if min_r_j < r_j:
                min_r_j = r_j
        R_i_l[l] = - min_r_j
    R_i = sum(R_i_l)

    return R_i


#--------------------------------#
# The achievable rate upbound of one stream of user i
# the K upbound of R_{i,l}
# carefully pass parameter into the function
# if index_eta(i,l,j) == None, the constraint don't exist.!!!
#--------------------------------#
def Rate_i_l_Bound(H, P, f, SNR,i,l,j):

    # compute the corresponding channel matrix
    H_j_i = H[j*N:(j+1)*N][:,i*N:(i+1)*N]
    # the column vector p
    p = np.array(P[index_I(i,l)*N:(index_I(i,l)+1)*N])
    print np.dot(H_j_i, np.expand_dims(p,axis=1))
    print 'dimension of multiplication:',(f * np.dot(H_j_i, np.expand_dims(p, axis=1))).size
    R_i_j = cvx.log(cvx.norm( f * np.dot(H_j_i, np.expand_dims(p, axis=1))) * SNR ) * np.log2(q)

    return R_i_j

# compute the augmented channel matrix \tilde(H)
# the function is a special case for given G and 6*6 H
def Augmented_chan_matrix(H,G):
    H_aug = np.array([])
    for j in range(K):
        for l in range(L_node[j]):
            supp_ind = index_S(j,l)
            if len(supp_ind) == 2:
                H_N = np.zeros((N, N*sum(L_node)))
                # print H_N[:,supp_ind[0]*N:(supp_ind[0] + 1)*N]
                # print np.array(H[j*N:(j+1)*N][:, index_inv(supp_ind[0])*N:(index_inv(supp_ind[0])+1)*N]) * float(G[index_I(j,l),supp_ind[1]])
                H_N[:,supp_ind[0]*N:(supp_ind[0] + 1)*N] = \
                np.array(H[j*N:(j+1)*N][:, index_inv(supp_ind[0])*N:(index_inv(supp_ind[0])+1)*N]) * float(G[index_I(j,l),supp_ind[1]])
                # print np.array(H[j * N:(j + 1) * N][:, index_inv(supp_ind[1]) * N:(index_inv(supp_ind[1]) + 1) * N]) * float(G[index_I(j, l), supp_ind[0]])
                H_N[:, supp_ind[1] * N:(supp_ind[1] + 1) * N] = \
                np.array(H[j * N:(j + 1) * N][:, index_inv(supp_ind[1]) * N:(index_inv(supp_ind[1]) + 1) * N]) * float(G[index_I(j, l), supp_ind[0]])
                H_aug = np.array(H_aug.tolist() + H_N.tolist())
    return H_aug

# compute the matrix that constraint receive shaping vector f_{j,l}
def F_constr_mat(H,P,j,l):
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
            H_p = np.array(H_p.tolist() + np.expand_dims(np.transpose(np.dot(H[j * N:(j + 1) * N][:, 0 : 1 * N], P[index_I(0,ii)*N:(index_I(0,ii)+1)*N])),axis=0).tolist())

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

    return F_constr_matrix

#
def Bilinear_optimize(H,G):


    # Initialization the alternatively algorithm
    H_tilde = Augmented_chan_matrix(H, G)
    # find a feasible non-zero P in the nullspace of H_tilde
    [s, v, d] = np.linalg.svd(H_tilde)
    fea_P = d[-1, :]
    P_init = E_total * fea_P / np.linalg.norm(fea_P, 2)

    # compute the constraint matrix for f_{j,l}
    F_constr_matrix = F_full_constr_matrix(H,P_init)

    # apply cvxpy to solve the relaxed optimization problem

    # Define the variables
    P = cvx.Variable(N * sum(L_node), 1)
    F = [0]*sum(L_node)
    R_i_l = [0]*sum(L_node)
    for ii in range(sum(L_node)):
        F[ii] = cvx.Variable(1,N)
        R_i_l[ii] = cvx.Variable()
    #R_i_l = cvx.Variable(sum(L_node)) # the rate of all data stream

    # compute all constraints
    Constr = []

    # the constraints for R_i_l
    for i in range(K):
        for l in range(L_node[i]):
            for j in range(K):
                if index_eta(i,l,j) != None:
                    print index_I(i,l)
                    Constr.append(R_i_l[index_I(i,l)] <= Rate_i_l_Bound(H, P_init, F[index_I(i,l)], SNR,i,l,j))

    # the constraints for f_j_l
    for ii in range(sum(L_node)):
        Constr.append(F[ii] * F_constr_matrix[ii*N:(ii+1)*N] == 0)
        Constr.append(cvx.norm(F[ii],2) <= 1)

    # objective function
    Obj = cvx.Maximize(sum(R_i_l))

    # solve the maximization problem
    Prob = cvx.Problem(Obj, Constr)
    Prob.solve()
    for ii in range(sum(L_node)):
        F[ii] = F[ii].value
        R_i_l[ii] = R_i_l[ii].value

    return  Prob.value, F.value, R_i_l.value


if __name__ == '__main__':

    # set random seed
    np.random.seed()
    # random channel matrix from user 1 to BS1
    H = np.random.rand(6,6)
    print 'channel matrix:\n', H

    print Bilinear_optimize(H, G)
    
    #print np.dot(H_tilde, d[-1,:])
    if False:
        P = cvx.Variable(N*sum(L_node),1)
        obj = cvx.Minimize(1)
        constraints = [Augmented_chan_matrix(H,G) * P == 0, cvx.sum_squares(P) <= E_total]
        prob = cvx.Problem(obj, constraints)
        prob.solve()
        print P.value
