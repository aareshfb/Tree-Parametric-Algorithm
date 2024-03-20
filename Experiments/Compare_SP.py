# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:21:23 2023

@author: aares
"""
import numpy as np
import time
from Parametric import Para_Algo
from GenData import GenTriDiag
import pandas as pd
import os
from qpsolvers import solve_qp
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
import networkx as nx
import multiprocessing as mp


def mu(Q,c,x): #obj without sum(lam)
    v=1/2*x@Q@x+c@x
    return(v)

def TDMAsolver(Ta, Tb, Tc, Td):
    Tnf = len(Td) 
    Tac, Tbc, Tcc, Tdc = map(np.array, (Ta, Tb, Tc, Td)) 
    for Tit in range(1, Tnf):
        Tmc = Tac[Tit-1]/Tbc[Tit-1]
        Tbc[Tit] = Tbc[Tit] - Tmc*Tcc[Tit-1] 
        Tdc[Tit] = Tdc[Tit] - Tmc*Tdc[Tit-1]
    Txc = Tbc
    Txc[-1] = Tdc[-1]/Tbc[-1]
    for Til in range(Tnf-2, -1, -1):
        Txc[Til] = (Tdc[Til]-Tcc[Til]*Txc[Til+1])/Tbc[Til]
    return Txc

def ShortestPathAlgo(Q,c,lam): #Algo for shortest path
    n=len(c)
    temp=-c[0]/Q[0,0]
    if 1/2*Q[0,0]*temp**2+c[0]*temp<-lam[0]:
        temp=-c[0]/Q[0,0]
    else:
        temp=0
    # x=[[]]
    # x[0].append(temp)
      
    v=[[np.nan]]
    f=5*np.ones(n)
    f[0]=1/2*Q[0,0]*temp**2+c[0]*temp+lam[0]*(1*(temp!=0))
    # xs=[np.array(temp)] #square brackets added here
    for m in range(1,n):
        v.append([])
        # x.append([])
        #f.append([])
        
        #f[m]=v[m]+sum(lam[:m+1])
        for k in range(m+1):
        
            Qt=Q[k:m+1,k:m+1]
            ct=c[k:m+1]
            # xt=solve_qp(Qt,ct,solver="gurobi")
            # xt=TDMAsolver(Qt,ct)
            xt=TDMAsolver(np.diagonal(Qt,-1),np.diagonal(Qt),np.diagonal(Qt,1),ct)
            v[m].append(mu(Qt,ct,xt)+sum(lam[k:m+1]))
            # if k==0:
            #     x[m].append(xt)
            # elif k==1:
            #     xtt=np.append(0,xt)
            #     x[m].append(xtt)
            # else:
            #     xtt=np.append(0,xt)
            #     x[m].append(np.append(xs[k-2],xtt))
            
    
        #f[m]=np.min(v[m]+np.append([0,0],f[:m-1]))
        #f[m]=min(f[m-1],0)
        f[m]=np.min(np.append(v[m],0)+np.append([0,0],f[:m]))
        idx=np.argmin(np.append(v[m],0)+np.append([0,0],f[:m]))
        
        # if idx==m+1:
        #     xs.append(np.append(xs[m-1],0))
        # else:
        #     xs.append(x[m][idx])
    #xs[-1] is the optimals
    # return(f,xs)
    return(f)


            
def Subshort(cp,Qp,lamp,pl):
    SSN = pl
    SSQ = [[0 for x in range(SSN+2)] for y in range(SSN+2)]
    SSc = [0 for x in range(SSN+2)]
    SSlam = [0 for x in range(SSN+2)]

    for i in range(1,SSN+1):
        SSc[i] = cp[i-1]
        SSlam[i] = lamp[i-1]
    for i in range(1,SSN):
        SSQ[i][i+1] = Qp[i-1][i]
        SSQ[i+1][i] = SSQ[i][i+1]
        SSQ[i][i] = Qp[i-1][i-1]
    SSQ[SSN][SSN] = Qp[SSN-1][SSN-1]
    SSQ[0][0] = 1
    SSQ[SSN+1][SSN+1] = 1
    SSl=[0 for x in range(SSN+2)]
    SSNprev = [0 for x in range(SSN+2)]
    for i in range(SSN+1):
        SSl[i+1]=10e+20
    for i in range(SSN+1):
        if SSl[i+1]>SSl[i]:
            SSl[i+1]=SSl[i]
            SSNprev[i+1] = i
        SSalbar = 0
        SSqbar = 10e+20
        SSkbar = 0
        for j in range(i+2,SSN+2):
            SSalbar = -SSc[j-1]-SSQ[j-2][j-1]/SSqbar*SSalbar
            SSqbar=SSQ[j-1][j-1]-SSQ[j-2][j-1]*SSQ[j-2][j-1]/SSqbar
            SSkbar = SSkbar-1/2*SSalbar*SSalbar/SSqbar+SSlam[j-1]
            if SSl[j]>(SSl[i]+SSkbar):
                SSl[j]=SSl[i]+SSkbar
                SSNprev[j]=i
    SSsx = [0.0 for x in range(SSN+2)]
    SSNN = SSNprev[SSN+1]
    i=SSNN
    j=SSN+1
    SSM=j-i+1
    if SSM>2:
        SSct = [0 for x in range(SSM-2)]
        SSQt = [[0 for x in range(SSM-2)] for y in range(SSM-2)]
        SSQtemp = SSQ[i:j+1][0:SSN+2]
        for k in range(SSM-2):
            SSct[k] = SSc[i+k+1]
            SSQt[k] = SSQtemp[k+1][i+1:j]
        TA = np.array(SSQt,float)
        TB = -np.array(SSct,float)
        Ta = TA.diagonal(-1)
        Tb = TA.diagonal()
        Tc = TA.diagonal(1)
        SSxt = TDMAsolver(Ta,Tb,Tc,TB)
        for k in range(SSNN+1,SSN+1):
            SSsx[k] = SSxt[k-SSNN-1]
    while SSNN != 0:
        SSNP = SSNprev[SSNN]
        i=SSNP
        j=SSNN
        SSM=j-i+1
        if SSM>2:
            SSct = [0 for x in range(SSM-2)]
            SSQt = [[0 for x in range(SSM-2)] for y in range(SSM-2)]
            SSQtemp = SSQ[i:j+1][0:SSN+2]
            for k in range(SSM-2):
                SSct[k] = SSc[i+k+1]
                SSQt[k] = SSQtemp[k+1][i+1:j]
            TA = np.array(SSQt,float)
            TB = -np.array(SSct,float)
            Ta = TA.diagonal(-1)
            Tb = TA.diagonal()
            Tc = TA.diagonal(1)
            SSxt = TDMAsolver(Ta,Tb,Tc,TB)
        for k in range(SSNP+1,SSNN):
            SSsx[k] = SSxt[k-SSNP-1]
        SSNN=SSNP
    SSsolux = SSsx[1:SSN+1]
    return SSsolux 
            
 
# filename='Experiments/TriDiag_Experiments2.csv'
# N=[20000]*9
# N.sort()
# for n in N:
#     Q=GenTriDiag(n)
#     lam=2.5*np.ones(n)
#     c=20*(np.random.rand(n)-0.5)
#     t_para_start = time.perf_counter()
#     opt_p,x_p=Para_Algo(Q, c, lam)
#     t_para_stop = time.perf_counter()
#     tPara=t_para_stop-t_para_start
#     # print(t_para_stop-t_para_start)
    
#     t_sp_start = time.perf_counter()
#     # f_sp=ShortestPathAlgo(Q, c, lam)
#     xsp=Subshort(c,Q,lam,n)
#     f_sp=mu(Q,c,np.array(xsp))+1*(np.array(xsp)!=0)@lam
#     t_sp_stop = time.perf_counter()
#     tSP=t_sp_stop-t_sp_start
#     # print(t_para_stop-t_para_start)
#     print(n)
#     Non_zeros=np.count_nonzero(x_p)
#     data={'n':n,'Number_of_non_zeros':Non_zeros,'lam_val':lam[0],
#           'Para_Algo_obj':opt_p,'Para_time':tPara,
#                 'Shortest_path_obj':f_sp,'Shortest_path_time':tSP}
#     df = pd.DataFrame.from_dict(data,orient='index' ,columns=['0'])
#     df=df.transpose()
#     df.to_csv(filename, mode='a', header=not os.path.isfile(filename))

def Get_Tree_Depth(Q):
    G=nx.from_numpy_array(Q)
    SP_dict=nx.shortest_path_length(G,0)
    Depth=np.max(list(SP_dict.values()))
    return(Depth)

def Run_Exp(Q,c,lam,filename,nL):
    n=len(c)
    Q_sparse=csc_matrix(Q)
    try:
        M=np.linalg.norm(c,2)/np.real(eigs(Q_sparse, k=1, which='SR')[0])
        M=M[0]
    except:
        M=np.linalg.norm(c,2)/np.min(np.linalg.eigvals(Q))
    print('Variable bounds:',M)
    t_para_start = time.perf_counter()
    opt_p,x_p=Para_Algo(Q, c, lam,M)
    t_para_stop = time.perf_counter()

    
    t_sp_start = time.perf_counter()
    f_sp=ShortestPathAlgo(Q, c, lam)
    xsp=Subshort(c,Q,lam,n)
    f_sp=mu(Q,c,np.array(xsp))+1*(np.array(xsp)!=0)@lam
    t_sp_stop = time.perf_counter()
    tSP=t_sp_stop-t_sp_start

    Non_zeros=np.count_nonzero(x_p)
    # print('Gurobi :',g_sol)
    print('Parametric Sol :',opt_p)
    Depth=Get_Tree_Depth(Q)
    data={'n':n,'Levels':nL,'Depth':Depth,'Number_of_non_zeros':Non_zeros,'lam_val':lam[0],
          'Para_Algo_obj':opt_p,'Para_time':t_para_stop-t_para_start,
                           'Shortest_path_obj':f_sp,'Shortest_path_time':tSP,'M':M}
    df = pd.DataFrame.from_dict(data,orient='index' ,columns=['0'])
    df=df.transpose()
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename))
    
    
# import networkx as nx
if __name__ == "__main__":

    File_Path='Results/'
    FileName='SP.csv'    
    lam_val=[7.5]*1
    N=[100]*1#,700,1000,2000,3000,4000,5000
    nL=[None]*len(N)
    # N=sorted(N*2)
    Q={}
    c={}
    lam={}
    print('Results saved in:',File_Path+FileName)
    for i in range(len(N)): #make range 11:
        n=N[i]
        
        c[i]=20*(np.random.rand(n)-0.5)
        lam[i]=lam_val[i]*np.ones(n)
        
        Qt=GenTriDiag(n)
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

