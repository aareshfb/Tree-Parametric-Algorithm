# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:37:44 2023

@author: aares
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:31:14 2023

@author: aares
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from Parametric import Para_Algo
import time
#from Verify2 import Solve_using_GP,Verification
import pandas as pd

accData=np.loadtxt('accelerometer.csv',delimiter=',',skiprows=1)
sigS=accData[:,0]
# sig=accData[:,1]
del accData

def GenCoeff(sig,mu,d):
    n=len(sig)
    diag=d+2*mu
    offdiag=-1*mu
    Q = np.zeros((int(n/d), int(n/d)))
    np.fill_diagonal(Q, diag)
    np.fill_diagonal(Q[1:], offdiag)
    np.fill_diagonal(Q[:, 1:], offdiag)
    c=-2*sum_d(sig,d)
    Q[0,0]=d+mu
    Q[-1,-1]=d+mu
    return(Q,c)

def sum_d(lst,d):
    return (np.array([sum(lst[i:i+d]) for i in range(0, len(lst), d)]))
            
def Get_c(c,d):
    cx=sum_d(c,d)
    result=[]
    for i in range(0,len(c),d):
        if i//d<len(cx):
            result.append(cx[i//d])
        result.extend(c[i:i+d])
    return(np.array(result))

def GenCoeff2(sig,mu,d,lamx,lamw):
    e=1
    n=len(sig)
    A=np.zeros((d+1,d+1))
    # diag=d*(1+2*mu)
    diag=e**2*np.ones(d+1)
    # offdiag=2
    np.fill_diagonal(A, diag)
    A[0,1:]=-e*1+A[0,1:]
    A[1:,0]=-e*1+A[1:,0]
    A[0,0]=d+2*mu
    blocks=[A]*int(n/d)
    Q=block_diag(*blocks)
    arr = np.zeros(int(n*(d+1)/d)-d-1)
    arr[::d+1] = -1*mu
    # B=np.diag(arr,d)
    Q=Q+np.diag(arr,d+1)+np.diag(arr,-d-1)
    # del B
    # c1=-2*sig
    # c1[::d]=0
    c=2*sig #
    lam=lamw*np.ones(int(n*(d+1)/d))
    lam[::d+1]=lamx
    c1=Get_c(e*c,d)
    c1[::d+1]=1/e*-c1[::d+1]
    return(Q,c1,lam)

n=len(sigS)
mu=0.5
fname='Results\\'
D=[10]
lamx=[400]
lamw=[150]

X={}

folder='Results\\'
filename='Experiment1.csv'#for csv
# code for non-robust
for d in D:
    timestamp=d*np.arange(0,n/d)
    for lx in lamx:
        lam=lx*np.ones(n)
        Q,c=GenCoeff(sigS, mu,d)
        t1=time.perf_counter()
        f,x=Para_Algo(2*Q, c, lam)
        t2=time.perf_counter()
        print(t2-t1)
        plt.figure(1)
        plt.plot(sum_d(sigS/d,d),label='Avg_Signal')
        plt.plot(x,label='x')
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.legend()
        plt.grid()
        # commented line to save picture
        # plt.savefig(fname+'Signal_'+str(int(lam[0]))+'no_w'+'_d'+str(d))
        plt.close()
        X['no_w'+str(lx)]=x
# code for robust
for d in D:
    timestamp=d*np.arange(0,n/d)
    for lx in lamx:
        for lw in lamw:
            Q,c,lam=GenCoeff2(sigS, mu,d,lx,lw)
            t1=time.perf_counter()
            f,x=Para_Algo(2*Q, c, lam)
            t2=time.perf_counter()
            del Q
            print(t2-t1)
            plt.figure(2)
            plt.plot(sigS,label='Original_Signal')
            plt.plot(timestamp,x[::d+1],label='x')
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.legend()
            plt.grid()
            X['timestamp']=timestamp
            X[str(lx)+'_'+str(lw)]=x[::d+1]
            # plt.savefig('Figures_4\\Signal_'+str(lx)+'_'+str(lw)+'_d'+str(d)) old stuff
            # plt.close()
            
            # plt.figure(3)
            # z=1*(x!=0)
            # zs=sum_d(z,d+1)
            # zs=zs-z[::d+1]0
            # xS=np.multiply(x[::d+1],zs)
            # plt.plot(timestamp,sum_d(sigS/d,d),label='Sig avg')
            # plt.plot(timestamp,xS,label='xS')
            # plt.legend()
            # plt.grid()
            
            plt.figure(4)
            plt.plot(timestamp,sum_d(sigS/d,d),label='Sig avg')
            plt.plot(timestamp,x[::d+1],label='x')
            plt.legend()
            plt.grid()
            # commented line to save picture
            # plt.savefig(fname+'Signal_'+str(lx)+'_'+str(lw)+'_d'+str(d))
            # plt.close()
            
            
            
# X['d']=D
# X['mu']=mu
df=pd.DataFrame(X)
df.to_csv(fname+'Vary_d2'+'.csv',sep=',',header=True)
