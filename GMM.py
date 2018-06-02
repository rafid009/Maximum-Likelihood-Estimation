# -*- coding: utf-8 -*-
"""
Created on Sun May  6 18:41:47 2018

@author: ASUS
"""

import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.datasets import make_spd_matrix
from scipy.stats import  multivariate_normal
from matplotlib.patches import Ellipse
from scipy.stats import chi2

MIN_LIMIT = -999999
MIN_DIFF = 0.0000000001

def plot_cov(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)

def generateZ(d, k):
    Z = []
    mu = np.array([[0.3,4],[5.5,0.25],[7,7]])
    for i in range(k):
        #z_sym = np.random.rand(d,d)
        #z = (np.tril(z_sym) + np.tril(z_sym).T)/2
        seed = i+10
        z = make_spd_matrix(d, random_state=seed) #(z_sym + z_sym.T)/2
        #mu = np.random.rand(d)
        Z.append((mu[i], z))
    return Z


def get_E_step(X, Mu, Cov, w, d, k, N):
    P = []
    for i in range(k):

        p_ij = multivariate_normal.pdf(X, Mu[i], Cov[i])
#        print("pij =")
#        print(p_ij)
        p_ij = p_ij*w[i]
        P.append(p_ij)
    P = np.array(P).T
    weight = np.zeros(N)
    for j in range(N):
        weight[j] = np.sum(P[j])
        P[j] = P[j]/weight[j]

#    print(P.shape)

    return P.copy(), weight.copy()






def get_all_mu(X, P, k, N):
    Mu = []
    for i in range(k):
#        print("In mu P: ",P[:,i].shape)
        u = np.dot(np.reshape(P[:,i], (N,1)).T, X)
#        print("In mu u: ",u.shape)
#        print("u_size = ", X.T.shape, np.reshape(P[:,i], (N,1)).shape)
        u_i =u[0]/np.sum(P[:,i])
#        print("u_i: ",u_i)
        Mu.append(u_i)
    return np.array(Mu).copy()

def get_all_sigma(X, Mu, P, N, k, d):
    Cov = []
    p_temp = P.T
    for i in range(k):
#        print(X.shape)
#        print(Mu.shape)
#        print((X-Mu[i]).shape)
        diff = X - Mu[i]
#        print("diff : ",diff.shape)
        new_shape = np.reshape(p_temp[i], (1, N))
#        print(new_shape.shape)

        temp = new_shape*diff.T
        sig = np.dot(temp, diff)
        sig /= np.sum(p_temp[i])

#        print("P.T: ",p_temp.shape)
#        print(sig.shape)

        Cov.append(sig)
    return np.array(Cov).copy()

def get_all_w(P, N, k):
    w = []
    for i in range(k):
        weight = np.sum(P[:,i])/N
        w.append(weight)
    w = np.array(w)
    w /= np.sum(w)
#    print("W: \n",w)
    return w.copy()

def get_log_likelihood(X, w, weight, Mu, Cov, N, P, k, d):
    sum_log = 0
    #temp = np.zeros(N).T
#    print("P_sum: ",np.sum(P))
#    tP_ij = P*np.sum(P)
    for j in range(N):
        temp = np.sum(P[j]*w)#weight[j]
#        temp = weight[j]
        sum_log += np.log(temp)


    return sum_log



def EM_algo(X, k, Mu, Cov, w, d, N):
    prev_like = MIN_LIMIT
    count = 0
#    loop = 0
    while True:#loop <= 100:
        count += 1
        P, weight = get_E_step(X, Mu, Cov, w, d, k, N)
#        print(count," Run: ")
#        print(P)
        Mu = get_all_mu(X, P, k, N)

#        print(Mu)
        Cov = get_all_sigma(X, Mu, P, N, k, d)
#        print(Cov)
        w = get_all_w(P, N, k)
        #    print(w)
        plt.clf()
        log_like = get_log_likelihood(X, w, weight, Mu, Cov, N, P, k, d)
        print(log_like)
        if np.abs(log_like - prev_like) <= MIN_DIFF:
            break;
        prev_like = log_like
        plt.scatter(X[:,0], X[:,1])
        plt.scatter(Mu[:,0], Mu[:,1])
        for i in range(k):
            plot_cov(Cov[i], Mu[i])
#        plt.draw()
        plt.show()
#        plt.contourf()


#        loop+=1



    print(Mu)
    print(Cov)
    print(w)






def main():
    N = 800
    d = 2
    np.random.seed(20)
    k = 3
    w = np.random.rand(k)
    w = w/np.sum(w)
    print(w)
    Z = generateZ(d, k)
    index = [i for i in range(k)]
    #print(index)
    print(Z)
    X = np.zeros(N)
    Y = np.zeros(N)
    #mu = []
    Z_idxs = []
    Xn = []
    for j in range(N):
        z_idx =np.random.choice(index, p=w)
        #print(z_idx)
        mu_idx, sigma_idx = Z[z_idx]
        #mu.append(np.random.rand())
        #cov.append(sigma_idx)
        Z_idxs.append(z_idx)
        X[j], Y[j] = np.random.multivariate_normal(mu_idx, sigma_idx, check_valid='warn')
        #print("x = ",X[j],", y = ",Y[j])
        x_j = np.array([X[j], Y[j]])
        Xn.append(x_j)
    plt.scatter(X, Y)
    Mu = np.random.random((k, d))
    print(Mu)
    Cov = []

    for i in range(k):

        seed = i+10
        z = make_spd_matrix(d, random_state=seed) #(z_sym + z_sym.T)/2
        #mu = np.random.rand(d)
        Cov.append(z)

#    print(Cov)
    Cov = np.array(Cov)
    print(Cov)
    w = np.random.random(k)
#    print(w)
    w = w/np.sum(w)
    print(w)
    Xn = np.array(Xn)
#    print(Xn)
    start_time = time.clock()
#    get_E_step(Xn, Mu, Cov, w, d, k, N)
    EM_algo(Xn, k, Mu, Cov, w, d, N)
    end_time = time.clock()
#    print(log_like)
    print(end_time-start_time,"s")








if __name__ == '__main__':
    main()