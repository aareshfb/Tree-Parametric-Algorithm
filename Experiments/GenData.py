# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:49:28 2023

@author: aares
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
# from hierarchy_pos import hierarchy_pos
import random


def generate_numbers(n,l):
    numbers = []
    s = 0
    for i in range(l-1):
        # if i==0:
        #     num = random.randint(2,n/2-2)
        # else:
        #     try:
        #         num = random.randint(2,n - s-2)
        #     except:
        #         continue
        num = random.randint(4,int(n/l))
        numbers.append(num)
        s += num
    numbers.append(n - s)
    return np.array(numbers)

def GenDenseTree2(n,l):
    # split=np.random.randint(l+1,n-l-1,l)
    split=generate_numbers(n, l)
    Y=sparse.coo_matrix((0,0))
    # print(split)
    for s in split:
        Q=GenTree(s,np.random.randint(2,np.ceil(s/2)+1))
        Y=sparse.block_diag((Y,Q))
    Y1=Y.toarray()
    a=0
    for s in split[:-1]:
        a=a+s
        Y1[a,a-1]=-0.01
        Y1[a-1,a]=-0.01
    return(Y1)


def GenDenseTree_Sparse(n):
    G = nx.random_tree(n) #,seed=0
    # pos=hierarchy_pos(G,0) 
    # nx.draw(G, pos=pos, with_labels=True)
    Q=sparse.lil_matrix(n,n)
    for i in G.edges:
        Q[i]=-np.random.rand(1)
        Q[i[1],i[0]]=-np.random.rand(1)
    for i in range(n):
        Q[i,i]=1+np.abs(np.sum(Q[i,:]))
    Q=0.5*(Q+Q.T)

    return(Q)
def GenDenseTree_random(n):
    G = nx.random_tree(n) #,seed=0
    # pos=hierarchy_pos(G,0) 
    # nx.draw(G, pos=pos, with_labels=True)
    Q=np.zeros([n,n])
    for i in G.edges:
        Q[i]=-np.random.rand(1)
        Q[i[1],i[0]]=-np.random.rand(1)
    for i in range(n):
        Q[i,i]=1+np.abs(np.sum(Q[i,:]))
    Q=0.5*(Q+Q.T)

    return(Q)

def GenDenseTree(n,nL=None):
    if nL==None:
        return(GenDenseTree_random(n))
    else:
        return(GenDenseTree2(n,nL))

def GenTriDiag(n):
    # first diagonal below main diag: k = -1
    
    a = -(0.00+np.random.rand(n-1))
    # main diag: k = 0
    b = np.random.rand(n) 
    # first diagonal above main diag: k = 1
    # sum all 2-d arrays in order to obtain A
    Q = np.diag(a, k=-1) + np.diag(b, k=0) + np.diag(a  , k=1)
    Q=Q+Q.T
    Q=Q+np.eye(n)*(np.abs(np.max(np.linalg.eigvals(Q)))+5)
    return(Q)

def smallest_non_zero(matrix):
    non_zero_elements = matrix[matrix != 0]
    if non_zero_elements.size == 0:
        return None
    else:
        return np.min(np.abs(non_zero_elements))
    
def GenTree(n,k):
    Q=GenTriDiag(n)
    b=np.random.choice(range(2,n), size=k, replace=False, p=None)
    # print(b)
    for i in range(k):
        a=b[i]
        Q[0,a]=Q[a,a-1]
        Q[a,0]=Q[a,a-1]
        #Q[a,a+1]=0
        #Q[a+1,a]=0
        Q[a,a-1]=0
        Q[a-1,a]=0
        #print(a)
    return(Q)

def GenMTree(n,k,k1):
    F= True
    while F:
        Q=GenTriDiag(n)
        b=np.random.choice(range(k1,n), size=k1*k, replace=False, p=None)
        for j in range(k1):
            for i in range(k):
                a=b[j*k1+i]
                Q[2*j,a]=Q[a,a-1]
                Q[a,2*j]=Q[a,a-1]
                #Q[a,a+1]=0
                #Q[a+1,a]=0
                Q[a,a-1]=0
                Q[a-1,a]=0
    
    
        G=nx.Graph(Q)
        G.remove_edges_from(nx.selfloop_edges(G))
        F= not nx.is_tree(G)
        
    # pos = hierarchy_pos(G1,0)    
    # nx.draw(G1, pos=pos, with_labels=True)
    return (Q)
    
def DrawGraph(Q,flag=0,flag2='replace'):
    if flag2=='replace':
        plt.figure(1)
    else:
        plt.figure()    
    plt.clf()
    if flag==0:
        Qs=abs(Q)
        G=nx.Graph(Qs)
        G.remove_edges_from(nx.selfloop_edges(G))
        pos=nx.kamada_kawai_layout(G)
        nx.draw_networkx(G,pos)
    elif flag==1:
        G=nx.Graph(Q)
        G.remove_edges_from(nx.selfloop_edges(G))
        pos=nx.planar_layout(G)
        nx.draw_networkx(G,pos)
    elif flag==2:
        G=nx.Graph(Q)
        G.remove_edges_from(nx.selfloop_edges(G))
        nx.draw_networkx(G)
    plt.show()