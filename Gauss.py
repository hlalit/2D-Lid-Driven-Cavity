import numpy as np 
import bcs as bc 

def gauss_elim(A,B,P):

	N = int(bc.N)
	M = int(bc.M) 
	Q = np.zeros([N*M,N*M + 1], dtype = np.float)
	X = np.zeros([N*M,1], dtype = np.float)
	
	Q = np.concatenate((A,B),axis = 1)

	for j in range(0,N*M-1):	#converts the matrix into upper triangular form
		for i in range(j+1,N*M):
			Q[i][:] = Q[i][:] - (Q[i][j]/Q[j][j])*Q[j][:]

	for i in range(N*M-1,-1,-1):
		if i == N*M-1:
			X[i] = Q[i][N*M]/Q[i][i]
			
		elif i != 0:
			s = 0
			for k in range(i+1,N*M):
				s = s + Q[i][k]*X[i+1]

			X[i] = (Q[i][N*M] - s)/Q[i][i]	#found P/X as a column vector


	for i in range(0,N):
		for j in range(0,M):
			P[i][j] = X[i*(bc.M) + j]

	return 
