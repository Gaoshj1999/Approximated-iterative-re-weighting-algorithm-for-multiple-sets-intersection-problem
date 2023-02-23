# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 20:25:15 2022

@author: GAOSHIJIE
"""

from time import time
#from IRWA_2 import *
import numpy as np
import math
import matplotlib.pyplot as plt


def get_grad_x_k(H, g, x_k):
    return np.matmul(H.T, (np.matmul(H, x_k)-g))


def get_grad_w_x_k(w, Y):  # w is a vector(list), w: 1*N(cons_numb), Y:N*n
    size = len(w)
    w = np.asarray(w).reshape(size, 1) #w:N*1
    #print("Y.shape: ", Y.shape)
    #print("w.shape: ", w.shape)
    #print("Y.T.shape: ", (Y.T).shape)
    return np.matmul(Y.T, w)


# B is always a constant at x_t,B:N*n=[[b1^T],...,[bn^T]], P:n*N
def get_B(w, P):
    N = len(w)
    B = (w[0]*P[:, 0:1]).T
    if N != 1:
        for i in range(N-1):
            P_c_i = P[:, (i+1):(i+2)]
            b_i = w[i+1]*P_c_i  # b_i: n*1
            b_i = b_i.T  # b_i:1*n
            B = np.vstack((B, b_i))  # B: B=[[B],[b_i^T]]
    return B


def WAx_tilde(w, x_tilde):  # W: N*n
    N = len(w)  # cons_numb
    n = x_tilde.shape[0]
    w_i = np.asarray(w)
    WAx = np.tile(w_i, (n, 1))  # get W=[[w1,...,wN],[w1,...,wN]...]
    WAx = WAx.T  # W=[[w1,...,w1],..,[wN,...,wN]]
    print("x_tilde:",x_tilde)
    print("WAx:",WAx)
    # print("WAx:",WAx)
    for i in range(N):
        WAx[i] = WAx[i]*x_tilde[i]  # get WAx
    print("WAx:",WAx)
    print("WAx.T:",WAx.T)
    print("WAx:",WAx)
    return WAx


def x_k_next(H, g, w, x_k, Y_tilde, tau):
    grad_x_k = get_grad_x_k(H, g, x_k)
    w_x = get_grad_w_x_k(w, Y_tilde)
    x_next = x_k - tau*(w_x + grad_x_k)
    return x_next


def Y_k_next(B, w, Y_k, x_tilde, theta, lamb):
    #print("/////////")
    #print("Y_k:", Y_k)
    WAx = WAx_tilde(w, x_tilde)
    #print("WAx:", WAx)
    #print("WAx:",WAx)
    parm = 1/(1/theta+1/lamb)
    Y_next = parm*(WAx-B+(1/theta)*Y_k)
    #print("Y_next:", Y_next)
    return Y_next


def X_tilde(x_k, x_next):
    print("///////")
    print("x_k:",x_k)
    print("x_next:",x_next)
    print("return:",2*x_next-x_k)
    return 2*x_next-x_k


def y_tilde(Y_k):
    return Y_k


def primal_dual_algorithm(H, g, x_t, w, P, tau, theta, lamb, K):
    B = get_B(w, P)
    print(B)
    x_k = x_t #initialize x_k = x_t
    Y_k = np.zeros((P.shape[1], P.shape[0]))
    iteration = 0
    for i in range(K):
        iteration = i+1
        Y_tilde = y_tilde(Y_k)
        x_next = x_k_next(H, g, w, x_k, Y_tilde, tau)
        print("x_next:",x_next)
        x_tilde = X_tilde(x_k, x_next)
        Y_next = Y_k_next(B, w, Y_k, x_tilde, theta, lamb)
        if np.linalg.norm(x_k-x_next, ord=2) < 1e-2*0.5 and np.linalg.norm(Y_k-Y_next, ord='fro') < 1e-3:
            return x_next, iteration
        Y_k = Y_next
        x_k = x_next
    return x_k, iteration


H = np.array([[21, 31, 13], [33, 15, 92]])
g = np.array([[1], [1]])
x_t = np.array([[-0.2],
 [ 0.7],
 [ 0.1]])
P = np.array([[-0.34257426],
             [0.50693069],
              [0.08217822]])
w = [0.9722398628729891]

tau = 0.00001
theta = 0.00001
lamb = 20000
K = 1000

x_sol, iteration = primal_dual_algorithm(H, g, x_t, w, P, tau, theta, lamb, K)

print("x_sol:", x_sol)
print("iteration:", iteration)
