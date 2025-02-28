# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:46:36 2023

@author: aareshfb


This file runs experiments for Varying Lam where lam is a random number.
Here we compare against with gurobi using the perspective reformulation.
"""
import numpy as np
from GenData import GenDenseTree,GenTree,DrawGraph,smallest_non_zero
import multiprocessing as mp
from Verify import Solve_using_GP,Verification_P, Verification
from Para_algo2 import Para_Algo
import time
import pandas as pd
import os
import networkx as nx
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
import sys

def Get_Tree_Depth(Q):
    G=nx.from_numpy_array(Q)
    SP_dict=nx.shortest_path_length(G,0)
    Depth=np.max(list(SP_dict.values()))
    return(Depth)

def Run_Exp(Q,c,lam,filename,nL,LAM_VAL):
    n=len(c)
    Q_sparse=csc_matrix(Q)
    mu_min=np.real(eigs(Q_sparse, k=1, which='SR')[0])
    M=np.linalg.norm(c,2)/mu_min
    M=M[0]
    t_para_start = time.perf_counter()
    opt_p,x_p=Para_Algo(Q, c, lam,M)
    t_para_stop = time.perf_counter()

    g_sol,time_g,Gap_g,x_g,BnB_Nodes=Verification_P(Q-mu_min*np.identity(n),mu_min*np.ones(n), c, lam,M)
    x_g_np=np.array(x_g.values())
    # gurobi_true_obj=0.5*x_g_np.T@Q@x_g_np+x_g_np.T@c+lam.T@(x_g_np!=0)
    
    g_sol_M,time_g_M,Gap_g_M,x_g_M,BnB_Nodes_M=Verification(Q, c, lam,M)
    x_g_np_M=np.array(x_g_M.values())
    gurobi_cal_obj_M=0.5*x_g_np_M.T@Q@x_g_np_M+x_g_np_M.T@c+lam.T@(x_g_np_M!=0) #
    
    Non_zeros=np.count_nonzero(x_p)
    # print('Gurobi :',g_sol)
    print('Parametric Sol :',opt_p)
    Depth=Get_Tree_Depth(Q)
    data={'n':n,'Levels':nL,'Depth':Depth,'Number_of_non_zeros':Non_zeros,'lam_val':LAM_VAL,'Gurobi_obj_P':g_sol,
          'Para_Algo_obj':opt_p,'Gurobi_opt_gap_P':Gap_g,
          'Gurobi_time_P':time_g,'ST_Para_time':t_para_stop-t_para_start,
          'BnB_Nodes_P':BnB_Nodes,'M':M,'Gurobi_obj_M':g_sol_M,'Calculated_Gurobi_obj_M':gurobi_cal_obj_M,
          'Gurobi_opt_gap_M':Gap_g_M,'Objective_Diff_M':(opt_p-g_sol_M),'Error%_M':np.round((opt_p-g_sol_M)/np.abs(g_sol_M)*100,4),
          'Gurobi_time_M':time_g_M,'BnB_Nodes_M':BnB_Nodes_M}
    df = pd.DataFrame.from_dict(data,orient='index' ,columns=['0'])
    df=df.transpose()
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename))
    
    
# import networkx as nx
if __name__ == "__main__":

    File_Path='Computation_Results/'
    FileName='Vary_n_perspective.csv'
    no_exp=1
    size=int(sys.argv[1])
    LAM=float(sys.argv[2])
    # sigma=float(LAM)/4    
    lam_val=[LAM]*no_exp
    N=[size]*no_exp#,700,1000,2000,3000,4000,5000
    nL=[None]*len(N)
    # N=sorted(N*2)
    Q={}
    c={}
    lam={}
    for i in range(len(N)): #make range 11:
        n=N[i]
        
        c[i]=20*(np.random.rand(n)-0.5)
        lam[i]=lam_val[i]*np.ones(n)
        # lam[i]=np.random.uniform(LAM-sigma,LAM+sigma,n)
        Qt=GenDenseTree(n,nL[i])
        Qt=Qt-np.diag(np.diag(Qt))
        Qt=Qt+np.diag(1-np.sum(Qt,1))
        Q[i]=Qt #identy matrix for the perspective refomulation is added in Run_Exp
        
        Run_Exp(Q[i], c[i], lam[i], File_Path+FileName, nL[i],lam_val[i])
        
        # np.save(File_Path+'/Data/Q'+str(i),Q[i])
        # np.save(File_Path+'/Data/c'+str(i),c[i])

    # processes = [mp.Process(target=Run_Exp, args=(Q[p_id],c[p_id],lam[p_id],File_Path+FileName,nL[p_id])) for p_id in range(len(N))]
    # for process in processes:
        # process.start()
    # for process in processes:
        # process.join()
