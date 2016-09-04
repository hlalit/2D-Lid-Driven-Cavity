import numpy as np
import bcs as bc 

#define a function that calculates the convective part 
def convection(us,vs,u,v,uold,vold,time):

    if time == 1:   #implement euler step
        for i in range(1,bc.N - 1):
            for j in range(1,bc.M - 1):                
                us[i][j] = bc.uo[i][j] - bc.dt*bc.uo[i][j]*(bc.uo[i][j+1] - bc.uo[i][j-1])/(2*bc.dx)
                us[i][j] = us[i][j] - bc.dt*bc.vo[i][j]*(bc.uo[i+1][j] - bc.uo[i-1][j])/(2*bc.dy)
                
            
                vs[i][j] = bc.vo[i][j] - bc.dt*bc.uo[i][j]*(bc.vo[i][j+1] - bc.vo[i][j-1])/(2*bc.dx)
                vs[i][j] = vs[i][j] - bc.dt*bc.vo[i][j]*(bc.vo[i+1][j] - bc.vo[i-1][j])/(2*bc.dy)  
                
    elif time == 2:  #implement adams bashforth step with u_0
        for i in range(1,bc.N - 1):
            for j in range(1,bc.M - 1):
                us[i][j] = u[i][j] - 1.5*bc.dt*(u[i][j]*(u[i][j+1] - u[i][j+1])/(2*bc.dx))
                us[i][j] = us[i][j] - 1.5*bc.dt*(v[i][j]*(u[i+1][j] - u[i-1][j])/(2*bc.dy))
                us[i][j] = us[i][j] + 0.5*bc.dt*(bc.uo[i][j]*(bc.uo[i][j+1] - bc.uo[i][j-1])/(2*bc.dx))
                us[i][j] = us[i][j] + 0.5*bc.dt*(bc.vo[i][j]*(bc.uo[i+1][j] - bc.uo[i-1][j])/(2*bc.dy))
                
                vs[i][j] = v[i][j] - 1.5*bc.dt*(u[i][j]*(v[i][j+1] - v[i][j-1])/(2*bc.dx))
                vs[i][j] = vs[i][j] - 1.5*bc.dt*(v[i][j]*(v[i+1][j] - v[i-1][j])/(2*bc.dy))
                vs[i][j] = vs[i][j] + 0.5*bc.dt*(bc.uo[i][j]*(bc.vo[i][j+1] - bc.vo[i][j-1])/(2*bc.dx))
                vs[i][j] = vs[i][j] + 0.5*bc.dt*(bc.vo[i][j]*(bc.vo[i+1][j] - bc.vo[i-1][j])/(2*bc.dy))

    else:           #implement adams bashforth step with u_n-1
        for i in range(1,bc.N - 1):
            for j in range(1,bc.M - 1):
                us[i][j] = u[i][j] - 1.5*bc.dt*(u[i][j]*(u[i][j+1] - u[i][j-1])/(2*bc.dx))
                us[i][j] = us[i][j] - 1.5*bc.dt*(v[i][j]*(u[i+1][j] - u[i-1][j])/(2*bc.dy))
                us[i][j] = us[i][j] + 0.5*bc.dt*(uold[i][j]*(uold[i][j+1] - uold[i][j-1])/(2*bc.dx))
                us[i][j] = us[i][j] + 0.5*bc.dt*(vold[i][j]*(uold[i+1][j] - uold[i-1][j])/(2*bc.dy))
                 
                vs[i][j] = v[i][j] - 1.5*bc.dt*(u[i][j]*(v[i][j+1] - v[i][j-1])/(2*bc.dx))
                vs[i][j] = vs[i][j] - 1.5*bc.dt*(v[i][j]*(v[i+1][j] - v[i-1][j])/(2*bc.dy))
                vs[i][j] = vs[i][j] + 0.5*bc.dt*(uold[i][j]*(vold[i][j+1] - vold[i][j-1])/(2*bc.dx))
                vs[i][j] = vs[i][j] + 0.5*bc.dt*(vold[i][j]*(vold[i+1][j] - vold[i-1][j])/(2*bc.dy))                                 

    us[bc.N-1][:] = 1.0  #ensuring that top boundary condition is met before return
    # print(us)
    return
    
            
