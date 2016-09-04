import numpy as np

L = float(1.0)
Re = float(10) 		#reynolds number 
N = int(11)  		#mesh cells in x-direction
M = int(11)  		#mesh cells in y-direction
CFL = float(0.8)
umax = float(1)
uo = np.zeros([N,M], dtype = np.float)
vo = np.zeros([N,M], dtype = np.float)

dx = L/(N-1)
dy = L/(M-1)
dt = CFL*dx/umax

uo[N-1][:] = 1.0 	#top side u velocity = 1
