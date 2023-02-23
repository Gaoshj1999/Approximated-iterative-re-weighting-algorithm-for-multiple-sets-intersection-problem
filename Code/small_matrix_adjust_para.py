# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:34:56 2022

@author: GAOSHIJIE
"""
from time import time
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import math
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
    #print("shape:",x_tilde.shape)
    N = len(w)  # cons_numb
    n = x_tilde.shape[0]
    w_i = np.asarray(w)
    WAx = np.tile(w_i, (n, 1))  # get W=[[w1,...,wN],[w1,...,wN]...]
    WAx = WAx.T  # W=[[w1,...,w1],..,[wN,...,wN]]
    #print("x_tilde:",x_tilde)
    #print("WAx:",WAx)
    #print("WAx:",WAx)
    for i in range(N):
        #print("WAx[i]:", WAx[i].shape)
        #print("x_tilde[i]", x_tilde[i].shape)
        WAx[i] = WAx[i]*x_tilde[i]  # get WAx
    #print("WAx:",WAx)
    #print("WAx.T:",WAx.T)
    #print("WAx:",WAx)
    return WAx


def x_k_next(H, g, w, x_k, Y_tilde, tau):
    grad_x_k = get_grad_x_k(H, g, x_k)
    w_x = get_grad_w_x_k(w, Y_tilde)
    #print(w_x)
    #print("tau:",tau)
    x_next = x_k - tau*(w_x + grad_x_k)
    #print("x_next:",x_next)
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
    #print("///////")
    #print("x_k:",x_k)
    #print("x_next:",x_next)
    #print("return:",2*x_next-x_k)
    return 2*x_next-x_k


def y_tilde(Y_k):
    return Y_k


def primal_dual_algorithm(H, g, x_t, w, P, tau, theta, lamb, K):
    B = get_B(w, P)
    #print(B)
    x_k = x_t #initialize x_k = x_t
    Y_k = np.zeros((P.shape[1], P.shape[0]))
    iteration = 0
    for i in range(K):
        iteration = i+1
        Y_tilde = y_tilde(Y_k)
        x_next = x_k_next(H, g, w, x_k, Y_tilde, tau)
        #print("x_next:",x_next)
        x_tilde = X_tilde(x_k, x_next)
        Y_next = Y_k_next(B, w, Y_k, x_tilde, theta, lamb)
        if np.linalg.norm(x_k-x_next, ord=2) < 1e-2 and np.linalg.norm(Y_k-Y_next, ord='fro') < 1e-2:
            return x_next
        Y_k = Y_next
        x_k = x_next
    return x_k      
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
        obj_value.append(0.5*np.linalg.norm(np.matmul(H, x_next)-g, ord=2)**2)
        step_t = 1/(12500*math.sqrt(i+1)) 
        for j in range(cons_numb):
            if np.linalg.norm(x_next-x_t, ord=2) < M*(Varpsilon[j])**(1+gamma):
                Varpsilon[j] = Varpsilon[j]*eta[j]
        if np.linalg.norm(x_next-x_t, ord=2) < delta[0] and np.linalg.norm(Varpsilon, ord=2) < delta[1]:
            return x_next, X, obj_value
        x_t = x_next
        #print("step_t:", step_t)
    return x_t, X, obj_value
   
def IRWA_accelerate(H, g, A, B, cons_numb, x_0, step_t, lamb, gamma, Varpsilon, eta, delta, tau, theta, M, N):  
    x_t = x_0 #start point
    X = []
    obj_value = []
    X.append(x_0)
    obj_value.append(0.5*np.linalg.norm(np.matmul(H, x_0)-g, ord=2)**2)
    for i in range(N):
        #grad_t = get_grad_at_x_t(H, g, x_t)
        w = get_W(A, B, x_t, cons_numb, Varpsilon)
        P = get_P(A, B, x_t, cons_numb)
        x_next = primal_dual_algorithm(H, g, x_t, w, P, tau, theta, lamb, 40)
        #print("here:",x_next)
        X.append(x_next)
        obj_value.append(0.5*np.linalg.norm(np.matmul(H, x_next)-g, ord=2)**2)
        for j in range(cons_numb):
            if np.linalg.norm(x_next-x_t, ord=2) < M*(Varpsilon[j])**(1+gamma):
                Varpsilon[j] = Varpsilon[j]*eta[j]
        if np.linalg.norm(x_next-x_t, ord=2) < delta[0] and np.linalg.norm(Varpsilon, ord=2) < delta[1]:
            return x_next, X, obj_value
        x_t = x_next
    return x_t, X, obj_value
         
def QP_problem(cons_numb, m, n, x_0, step_t, lamb, gamma, Varpsilon, eta, delta, tau, theta, M, N): 
    #problen construction   
    '''
    H = np.array([[21,31,13],[33,15,92]])
    g = np.array([[1],[1]])
    A = []
    A.append(np.array([[1,2,4],[2,3,2]]))
    B = []
    B.append(np.array([[1],[1]]))
    '''
    H, g = QP_objective(m, n) 
    A, B = QP_constraint(cons_numb, m, n) 
    #print(A)
    #print(B)
    #problem solving
    t_start=time()
    x_sol, X, obj_value = IRWA(H, g, A, B, cons_numb, x_0, step_t, lamb, gamma, Varpsilon, eta, delta, M, N)
    t_end=time()
    time_1 = t_end-t_start
    print("time1 :", time_1)
    t_start=time()
    x_sol_2, X_2, obj_value_2 = IRWA_accelerate(H, g, A, B, cons_numb, x_0, step_t, lamb, gamma, Varpsilon, eta, delta, tau, theta, M, N)
    t_end=time()
    time_2 = t_end-t_start
    print("time2: ", time_2)
    f_min = 0.5*np.linalg.norm(np.matmul(H, x_sol)-g, ord=2)**2
    f_min_2 = 0.5*np.linalg.norm(np.matmul(H, x_sol_2)-g, ord=2)**2
    return x_sol, f_min, X, obj_value, x_sol_2, f_min_2, X_2, obj_value_2, time_2

#x_0 = np.ones((3,1))
m = 8
n = 32
N = 2

x_0 = np.zeros((n,1))
lamb = 100
gamma = 0.15
Varpsilon = np.ones(N)
eta = np.ones(N)*0.02
delta = [1e-4, 1e-5]
tau = 0.00001
theta = 0.00001
Tau = []
Theta = []
Iteration = []
Iteration_2 = []
Time_para_2 = []
for i in range(200):
    time_total_2 = 0
    tau += 0.000002
    Tau.append(math.log(tau,10))
    Theta.append(theta)
    for i in range(10):
        x_sol, f_min, X, obj_value, x_sol_2, f_min_2, X_2, obj_value_2, time_2 = QP_problem(N, m, n, x_0, 1, lamb, gamma, Varpsilon, eta, delta, tau, theta, 100, 5000)#step_t is defined in IRWA
        time_total_2 = time_total_2 + time_2
    time_avg_2 = time_total_2/10
    Time_para_2.append(time_avg_2)
'''
t_start=time()
x_sol, f_min, X, obj_value, x_sol_2, f_min_2, X_2, obj_value_2 = QP_problem(N, m, n, x_0, 1, lamb, gamma, Varpsilon, eta, delta, tau, theta, 100, 5000)#step_t is defined in IRWA
iteration = len(obj_value)
iteration_2 = len(obj_value_2)
t_end=time()

#print(x_sol)

print("f_min:", f_min)
print("IRWA iteration:", iteration)

#print(x_sol_2)
print("f_min_2:", f_min_2)
print("IRWA_acc iteration:", iteration_2)

t_cost=t_end-t_start
print("time cost: ", t_cost) 
'''

plt.plot(Tau, Time_para_2, label="AIRWA")
#plt.plot(obj_value_2, label="AIRWA")
plt.legend()
font_properties = fm.FontProperties("KaiTi")
plt.title('参数tau', fontproperties = font_properties)
plt.xlabel('log_10(tau)')
plt.ylabel('平均运行时长', fontproperties = font_properties)
plt.show()
