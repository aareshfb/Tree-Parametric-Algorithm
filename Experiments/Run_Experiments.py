# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:46:36 2023

@author: aares
"""
import numpy as np
from GenData import GenDenseTree,GenTree,DrawGraph,smallest_non_zero
import multiprocessing as mp
from Verify import Solve_using_GP,Verification
from Parametric import Para_Algo
import time
import pandas as pd
import os
import networkx as nx
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix

def Get_Tree_Depth(Q):
    G=nx.from_numpy_array(Q)
    SP_dict=nx.shortest_path_length(G,0)
    Depth=np.max(list(SP_dict.values()))
    return(Depth)

def Run_Exp(Q,c,lam,filename,nL):
    n=len(c)
    Q_sparse=csc_matrix(Q)
    M=np.linalg.norm(c,2)/np.real(eigs(Q_sparse, k=1, which='SR')[0])
    M=M[0]
    t_para_start = time.perf_counter()
    opt_p,x_p=Para_Algo(Q, c, lam,M)
    t_para_stop = time.perf_counter()

    g_sol,time_g,Gap_g,x_g,BnB_Nodes=Verification(Q, c, lam,M)

    Non_zeros=np.count_nonzero(x_p)
    # print('Gurobi :',g_sol)
    print('Parametric Sol :',opt_p)
    Depth=Get_Tree_Depth(Q)
    data={'n':n,'Levels':nL,'Depth':Depth,'Number_of_non_zeros':Non_zeros,'lam_val':lam[0],'Gurobi_obj':g_sol,
          'Para_Algo_obj':opt_p,'Gurobi_opt_gap':Gap_g,'Objective_Diff':(opt_p-g_sol),'Error%':np.round((opt_p-g_sol)/np.abs(g_sol)*100,4),'Gurobi_time':time_g,'ST_Para_time':t_para_stop-t_para_start,'BnB_Nodes':BnB_Nodes,'M':M}
    df = pd.DataFrame.from_dict(data,orient='index' ,columns=['0'])
    df=df.transpose()
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename))
    
    
# import networkx as nx
if __name__ == "__main__":

    File_Path='Results/'
    FileName='Vary_n.csv'    
    lam_val=[7.5]*1
    N=[100]*1#,700,1000,2000,3000,4000,5000
    nL=[None]*len(N)
    # N=sorted(N*2)
    Q={}
    c={}
    lam={}
    for i in range(len(N)): #make range 11:
        n=N[i]
        
        c[i]=20*(np.random.rand(n)-0.5)
        lam[i]=lam_val[i]*np.ones(n)
        
        Qt=GenDenseTree(n,nL[i])
        Qt=Qt-np.diag(np.diag(Qt))
        Qt=Qt+np.diag(1-np.sum(Qt,1))
        Q[i]=Qt

        # np.save(File_Path+'/Data/Q'+str(i),Q[i])
        # np.save(File_Path+'/Data/c'+str(i),c[i])

    processes = [mp.Process(target=Run_Exp, args=(Q[p_id],c[p_id],lam[p_id],File_Path+FileName,nL[p_id])) for p_id in range(len(N))]
    for process in processes:
        process.start()
    for process in processes:
        process.join()