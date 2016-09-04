import numpy as np
import bcs as bc 
import math

from conv import convection 
from pois import poisson
from visc import viscous

rms_u = float(0.0)
rms_v = float(0.0)
eps = float(1e-06)
time = int(1)

u = np.zeros([bc.N,bc.M], dtype = np.float)
v = np.zeros([bc.N,bc.M], dtype = np.float)
us = np.zeros([bc.N,bc.M], dtype = np.float)
vs = np.zeros([bc.N,bc.M], dtype = np.float)
uds = np.zeros([bc.N,bc.M], dtype = np.float)
vds = np.zeros([bc.N,bc.M], dtype = np.float)
uold = np.zeros([bc.N,bc.M], dtype = np.float)
vold = np.zeros([bc.N,bc.M], dtype = np.float)
P = np.zeros([bc.N,bc.M], dtype = np.float)


def RMS_E(u,v,uold,vold,time,rms_u,rms_v):
	
	if time == 1:
		rms_u = math.sqrt(sum(sum((u - bc.uo)**2)))
		rms_v = math.sqrt(sum(sum((v - bc.vo)**2)))

	else: 
		rms_u = math.sqrt(sum(sum((u - uold)**2)))
		rms_v = math.sqrt(sum(sum((v - vold)**2)))

	return (rms_u, rms_v)

while (time < 2):

    #call conv function
    convection(us,vs,u,v,uold,vold,time)
    uold = u
    vold = v

    #call poisson solver
    poisson(us,vs,P,uds,vds) 

    #call visc function
    viscous(uds,vds,u,v)   

    #calculate rms error at the end of each time step
    (rms_u,rms_v) = RMS_E(u,v,uold,vold,time,rms_u,rms_v)

    if (rms_u < eps) and (rms_v < eps):
    	print("The simulation as converged !!")
    	break
    else:
    	print(rms_u)
    	time = time + 1







