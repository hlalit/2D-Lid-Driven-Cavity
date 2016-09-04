import numpy as np
import bcs as bc 

from Gauss import gauss_elim

def poisson(us,vs,P,uds,vds):

    A = np.zeros([bc.N*bc.M, bc.N*bc.M], dtype = np.float)
    B = np.zeros([bc.N*bc.M,1], dtype = np.float)

    for i in range(0,bc.N):
        for j in range(0,bc.M):

            if i == 0 and j == 0:                    #botom left corner
                A[i*(bc.M) + j][i*(bc.M) + j] = -1/bc.dx**2 - 1/bc.dy**2 
                A[i*(bc.M) + j][(i+1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][i*(bc.M) + j+1] = 1/bc.dx**2                 
                B[i*(bc.M) + j] = (1/(bc.dt*bc.dx))*(us[i][j+1] - us[i][j])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (1/(bc.dt*bc.dy))*(vs[i+1][j] - vs[i][j])                
                
            elif i == 0 and j == bc.M - 1:           #bottom right corner 
                A[i*(bc.M) + j][i*(bc.M) + j] = -1/bc.dx**2 - 1/bc.dy**2 
                A[i*(bc.M) + j][(i+1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][i*(bc.M) + j-1] = 1/bc.dx**2                 
                B[i*(bc.M) + j] = (1/(bc.dt*bc.dx))*(us[i][j] - us[i][j-1])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (1/(bc.dt*bc.dy))*(vs[i+1][j] - vs[i][j]) 

            elif i == bc.N - 1 and j == 0:           #top left corner
                A[i*(bc.M) + j][i*(bc.M) + j] = -1/bc.dx**2 - 1/bc.dy**2 
                A[i*(bc.M) + j][(i-1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][i*(bc.M) + j+1] = 1/bc.dx**2                 
                B[i*(bc.M) + j] = (1/(bc.dt*bc.dx))*(us[i][j+1] - us[i][j])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (1/(bc.dt*bc.dy))*(vs[i][j] - vs[i-1][j]) 
            
            elif i == bc.N - 1 and j == bc.M - 1:    #top right corner 
                A[i*(bc.M) + j][i*(bc.M) + j] = -1/bc.dx**2 - 1/bc.dy**2 
                A[i*(bc.M) + j][(i-1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][i*(bc.M) + j-1] = 1/bc.dx**2                 
                B[i*(bc.M) + j] = (1/(bc.dt*bc.dx))*(us[i][j] - us[i][j-1])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (1/(bc.dt*bc.dy))*(vs[i][j] - vs[i-1][j]) 

            elif i == 0 and j > 0 and j < bc.M - 1:     #bottom side
                A[i*(bc.M) + j][i*(bc.M) + j] = -2/bc.dx**2 - 1/bc.dy**2 
                A[i*(bc.M) + j][(i+1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][i*(bc.M) + j+1] = 1/bc.dx**2   
                A[i*(bc.M) + j][i*(bc.M) + j-1] = 1/bc.dx**2                               
                B[i*(bc.M) + j] = (0.5/(bc.dt*bc.dx))*(us[i][j+1] - us[i][j-1])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (1/(bc.dt*bc.dy))*(vs[i+1][j] - vs[i][j]) 
            
            elif i == bc.N - 1 and j > 0 and j < bc.M - 1:  #top side 
                A[i*(bc.M) + j][i*(bc.M) + j] = -2/bc.dx**2 - 1/bc.dy**2 
                A[i*(bc.M) + j][(i-1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][i*(bc.M) + j+1] = 1/bc.dx**2   
                A[i*(bc.M) + j][i*(bc.M) + j-1] = 1/bc.dx**2                               
                B[i*(bc.M) + j] = (0.5/(bc.dt*bc.dx))*(us[i][j+1] - us[i][j-1])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (1/(bc.dt*bc.dy))*(vs[i][j] - vs[i-1][j]) 

            elif i > 0 and i < bc.N - 1 and j == 0:     #left side 
                A[i*(bc.M) + j][i*(bc.M) + j] = -1/bc.dx**2 - 2/bc.dy**2 
                A[i*(bc.M) + j][(i+1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][(i-1)*(bc.M) + j] = 1/bc.dy**2   
                A[i*(bc.M) + j][i*(bc.M) + j+1] = 1/bc.dx**2                               
                B[i*(bc.M) + j] = (1/(bc.dt*bc.dx))*(us[i][j+1] - us[i][j])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (0.5/(bc.dt*bc.dy))*(vs[i+1][j] - vs[i-1][j]) 

            elif i > 0 and i < bc.N - 1 and j == bc.M - 1:  #right side 
                A[i*(bc.M) + j][i*(bc.M) + j] = -1/bc.dx**2 - 2/bc.dy**2 
                A[i*(bc.M) + j][(i+1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][(i-1)*(bc.M) + j] = 1/bc.dy**2   
                A[i*(bc.M) + j][i*(bc.M) + j-1] = 1/bc.dx**2                               
                B[i*(bc.M) + j] = (1/(bc.dt*bc.dx))*(us[i][j] - us[i][j-1])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (0.5/(bc.dt*bc.dy))*(vs[i+1][j] - vs[i-1][j]) 

            elif i > 0 and i < bc.N - 1 and j > 0 and j < bc.M - 1: #inner nodes
                A[i*(bc.M) + j][i*(bc.M) + j] = -2/bc.dx**2 - 2/bc.dy**2 
                A[i*(bc.M) + j][(i+1)*(bc.M) + j] = 1/bc.dy**2 
                A[i*(bc.M) + j][(i-1)*(bc.M) + j] = 1/bc.dy**2                 
                A[i*(bc.M) + j][i*(bc.M) + j+1] = 1/bc.dx**2   
                A[i*(bc.M) + j][i*(bc.M) + j-1] = 1/bc.dx**2                               
                B[i*(bc.M) + j] = (0.5/(bc.dt*bc.dx))*(us[i][j+1] - us[i][j-1])
                B[i*(bc.M) + j] = B[i*(bc.M) + j] + (0.5/(bc.dt*bc.dy))*(vs[i+1][j] - vs[i-1][j]) 

    # print(A)

    gauss_elim(A,B,P) 
    
    solve_uds(us,vs,P,uds,vds)
    return 

def solve_uds(us,vs,P,uds,vds):
    
    for i in range(1,bc.N - 1): #interior nodes 
        for j in range(1,bc.M - 1): #interior nodes 
            uds[i][j] = us[i][j] - bc.dt*(P[i][j+1] - P[i][j-1])/(2*bc.dx)
            vds[i][j] = vs[i][j] - bc.dt*(P[i+1][j] - P[i-1][j])/(2*bc.dy)

    uds[bc.N-1][:] = 1.0

    return

