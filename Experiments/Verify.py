# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:35:35 2023

@author: aares
"""
import time
import gurobipy as gp
from gurobipy import GRB,quicksum
import numpy as np

def Verification(Q,c,lam,M): 
    """
    Parameters
    ----------
    Q : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    lam : TYPE
        DESCRIPTION.
    M: Variable bound
    Returns
    -------
    obj,time,MIP_gap,x

    """
    #Verify with Gurobi
    N=len(c)
    model = gp.Model("Verification")
    x = model.addVars(N,lb=-M,vtype=GRB.CONTINUOUS, name = "x")
    z = model.addVars(N,vtype=GRB.BINARY, name = "z")
    model.addConstrs(x[i]-M*z[i]<=0 for i in range(N))    # convert to big-M
    model.addConstrs(-x[i]-M*z[i]<=0 for i in range(N))    # convert to big-M
    obj =sum((c[k]*x[k]+lam[k]*z[k]) for k in range(N))+ sum((Q[k][l]*x[k]*x[l]/2) for k in range(N) for l in range(N))
    model.setParam("TimeLimit", 3600)
    # model.setParam("MIPGap", 0.01)
    # model.setParam("Threads", 1)
    model.setObjective(obj, GRB.MINIMIZE)
    
    t2_start = time.perf_counter()
    model.optimize()
    t2_stop = time.perf_counter() 
    print('Gurobi time: ',t2_stop-t2_start) 
    
    solutionz = model.getAttr('x', z)
    tempz = 0
    for i in range(N):
        if solutionz[i] != 0:
            tempz=tempz+1
    print('# non zero: ',tempz)
    obj1 = model.getObjective()
    print('\nGurobi Solution:',obj1.getValue())
    solutionx=model.getAttr('x', x)
    # print(solutionx)
    
    return(obj1.getValue(),t2_stop-t2_start,model.MIPGap,solutionx,model.NodeCount)

def Solve_using_GP(N_idx,Q,c,lam):
    Qsub=Q[np.ix_(N_idx,N_idx)]
    csub=c[N_idx]
    lamsub=lam[N_idx]
    _,_,_,xgurobi=Verification(Qsub, csub, lamsub)
    xg=np.array(list(xgurobi.values()))
    
    g_sol=0.5*xg@Qsub@xg+csub@xg+lamsub@(xg!=0)
    return(xg,g_sol)