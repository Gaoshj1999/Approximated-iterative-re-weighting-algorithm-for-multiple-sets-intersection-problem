# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:34:56 2022

@author: GAOSHIJIE
"""
from time import time
import numpy as np
import math
import matplotlib.pyplot as plt

#QP: 0.5||Hx-g||^2
#s.t  A_ix = b i=1,...N
#m>n

def get_random_matrix(lowbound, upbound, m, n):
    return np.random.uniform(lowbound, upbound, size=(m, n))
        
def get_random_vector(lowbound, upbound, m):
    return np.random.uniform(lowbound, upbound, size=(m, 1))

def QP_objective(m, n):   #get H:m*n and g:m*1
    return get_random_matrix(-10, 10, m, n), \
                 get_random_vector(-10, 10, m)            
        
def QP_constraint(cons_numb, m, n): #A_i: m*n, b:n*1
    A = [] #list of A_i
    B = [] #list of b_i 
    for i in range(cons_numb):    #A_iu_i=b_i
        A_i = get_random_matrix(-10, 10, m, n)
        u_i = get_random_vector(-1, 1, n)
        A.append(A_i)
        B.append(np.matmul(A_i, u_i)) 
   
    return A, B

def get_grad_at_x_t(H, g, x_t):
    return np.matmul(H.T, (np.matmul(H, x_t)-g))


'''
P_(C_i)(x_t)=argmin||x-x_t||_2, x \in C_i == A_ix=b_i
L(x,lamb)=0.5||x-x_t||_2^2+lamb^T(Ax-b)
x-x_t+A^Tlamb=0, Ax=b
AA^Tlamb=Ax_t-b
lamb=pseudo(AA^T)(Ax_t-b)
x=x_t-A^Tpseudo(AA^T)(Ax_t-b)
'''

def proj(A, b, x): #A is a matrix
    #print("A: ", A)
    #print("b: ", b)
    #print("x:", x)
    Apinv = np.linalg.pinv(np.matmul(A, A.T))
    lamb = np.matmul(Apinv, (np.matmul(A, x)-b))
    P_C_x = x - np.matmul(A.T, lamb)
    return P_C_x

def get_P(A, B, x_t, cons_numb): #P:[P_c_1,...,P_c_N], A is a list of matrix
    P = proj(A[0], B[0], x_t)
    if cons_numb != 1:
        for i in range(cons_numb-1):
            P_c_i_x = proj(A[i+1], B[i+1], x_t)
            P = np.concatenate((P, P_c_i_x), axis = 1)
    return P

#w_i^t=1/(\sqrt(||(x_t-P_C_i(x_t)||_2^2+varpsilon_i^2))

def get_w(A, b, x, varpsilon): #A is a matrix
    P_C_x = proj(A, b, x)
    return 1/(math.sqrt(np.linalg.norm(x-P_C_x)**2+varpsilon**2))

def get_W(A, B, x, cons_numb, Varpsilon): #A is a list of many matrix, B is a list of many vectors
    w = []
    for i in range(cons_numb):
        w_i = get_w(A[i], B[i], x, Varpsilon[i])
        w.append(w_i)
    return w
        
#(1/step_t+lambda\sum_(w_i^t))x=-grad_x_t+(1/step_t)x_t+lambda\sum_(w_i^t))P_C_i(x_t)  

def X_next(A, B, H, g, x_t, cons_numb, Varpsilon, step_t, lamb):
    HTH = np.matmul(H.T, H)
    Hg = np.matmul(H.T, g)
    parm = HTH
    sum_w = 0
    sum_w_and_p = 0
    for i in range(cons_numb):
        w_i = get_w(A[i], B[i], x_t, Varpsilon[i])
        #print("w_i: ", w_i)
        sum_w = sum_w + w_i
        sum_w_and_p = sum_w_and_p + w_i*proj(A[i], B[i], x_t)
    #print("sum_w: ", sum_w)
    #print("sum_w_and_p", sum_w_and_p)
    sum_w = sum_w*np.eye(H.shape[1])
    parm = parm + lamb*sum_w
    x_next = np.matmul(np.linalg.pinv(parm), Hg+lamb*sum_w_and_p)
    return x_next
    
#A is a list of many matrix, B is a list of many vectors
def IRWA(H, g, A, B, cons_numb, x_0, step_t, lamb, gamma, Varpsilon, eta, delta, M, N): 
    x_t = x_0 #start point
    X = []
    obj_value = []
    X.append(x_0)
    obj_value.append(0.5*np.linalg.norm(np.matmul(H, x_0)-g, ord=2)**2)
    for i in range(N):
        #grad_t = get_grad_at_x_t(H, g, x_t)
        x_next = X_next(A, B, H, g, x_t, cons_numb, Varpsilon, step_t, lamb)
        if i == 0:
            print("x_t:",x_t)
            print("P:", get_P(A, B, x_t, cons_numb))
            print("w:", get_W(A, B, x_t, cons_numb, Varpsilon))
            print("x_next:",x_next)
        '''
        if i == 1:
            print("x_next2:", x_next)
        '''
        X.append(x_next)
        '''
        if i < 10:
            print("x_next: ",x_next)
        '''
        obj_value.append(0.5*np.linalg.norm(np.matmul(H, x_next)-g, ord=2)**2)
        step_t = 1/(20*math.sqrt(i+1)) 
        for j in range(cons_numb):
            if np.linalg.norm(x_next-x_t, ord=2) < M*(Varpsilon[j])**(1+gamma):
                Varpsilon[j] = Varpsilon[j]*eta[j]
        if np.linalg.norm(x_next-x_t, ord=2) < delta[0] and np.linalg.norm(Varpsilon, ord=2) < delta[1]:
            return x_next, X, obj_value
        x_t = x_next
    return x_t, X, obj_value
        
def QP_problem(cons_numb, m, n, x_0, step_t, lamb, gamma, Varpsilon, eta, delta, M, N): 
    #problen construction      
    H = np.array([[21,31,13],[33,15,92]])
    g = np.array([[1],[1]])
    A = []
    A.append(np.array([[1,2,4],[2,3,2]]))
    B = []
    B.append(np.array([[1],[1]]))
    #print(A)
    #print(B)
    #problem solving
    x_sol, X, obj_value= IRWA(H, g, A, B, cons_numb, x_0, step_t, lamb, gamma, Varpsilon, eta, delta, M, N)
    f_min = 0.5*np.linalg.norm(np.matmul(H, x_sol)-g, ord=2)**2
    return x_sol, f_min, X, obj_value

#x_0 = np.ones((3,1))
x_0 = np.array([[-0.2],
 [ 0.7],
 [ 0.1]])
Varpsilon = [1]
eta = [0.2]
delta = [1e-4, 1e-5]
x_sol, f_min, X, obj_value = QP_problem(1, 2, 3, x_0, 1, 20000, 0.15, Varpsilon, eta, delta, 100, 5000)#step_t is defined in IRWA
iteration = len(obj_value)
print(x_sol)
print(f_min)
print(iteration)



plt.plot(obj_value)
plt.title('IRWA')
plt.xlabel('iteration')
plt.ylabel('objective')
plt.show()
