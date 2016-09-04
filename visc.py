import numpy as np 
import bcs as bc 

def viscous(uds,vds,u,v):
	
	N = int(bc.N)
	M = int(bc.M)

	uts = np.zeros([N,M], dtype = np.float)
	vts = np.zeros([N,M], dtype = np.float)
	A = np.zeros([(N-2)*(M-2),(N-2)*(M-2)], dtype = np.float)
	B = np.zeros([(N-2)*(M-2),1], dtype = np.float)
	X = np.zeros([(N-2)*(M-2),1], dtype = np.float)

	#sweep in the X-direction to calculate u_***
	for i in range(1,N-1):	#only internal nodes 
		for j in range(1,M-1):	#only internal nodes
			
			if j == 1:
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-1)] = (1/bc.dt + 1/(bc.Re*(bc.dx**2)))	
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j)] = -0.5/(bc.Re*(bc.dx**2))		
				B[(i-1)*(M-2) + (j-1)] = (0.5/(bc.Re*(bc.dx**2)))*uds[i][j+1] + (1/(bc.Re*(bc.dx**2)))*uds[i][j-1]
				B[(i-1)*(M-2) + (j-1)] = B[(i-1)*(M-2) + (j-1)] + (1/bc.dt - 1/(bc.Re*(bc.dx**2)))*uds[i][j]

			elif j == M-2:
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-1)] = (1/bc.dt + 1/(bc.Re*(bc.dx**2)))	
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-2)] = -0.5/(bc.Re*(bc.dx**2))		
				B[(i-1)*(M-2) + (j-1)] = (1/(bc.Re*(bc.dx**2)))*uds[i][j+1] + (0.5/(bc.Re*(bc.dx**2)))*uds[i][j-1]
				B[(i-1)*(M-2) + (j-1)] = B[(i-1)*(M-2) + (j-1)] + (1/bc.dt - 1/(bc.Re*(bc.dx**2)))*uds[i][j]
			
			elif j > 1 and j < M-2: 
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-1)] = (1/bc.dt + 1/(bc.Re*(bc.dx**2)))	
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j)] = -0.5/(bc.Re*(bc.dx**2))
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-2)] = -0.5/(bc.Re*(bc.dx**2))		
				B[(i-1)*(M-2) + (j-1)] = (0.5/(bc.Re*(bc.dx**2)))*uds[i][j+1] + (0.5/(bc.Re*(bc.dx**2)))*uds[i][j-1]
				B[(i-1)*(M-2) + (j-1)] = B[(i-1)*(M-2) + (j-1)] + (1/bc.dt - 1/(bc.Re*(bc.dx**2)))*uds[i][j]				

	tridiag(A,B,X,N,M)

	for i in range(1,N-1):
		for j in range(1,M-1):
			uts[i][j] = X[(i-1)*(M-2) + (j-1)]
	uts[N-1][:] = 1.0

	#sweep in the Y-direction to calculate u_n+1
	for j in range(1,M-1):	#only internal nodes 
		for i in range(1,N-1):	#only internal nodes
			
			if i == 1:
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-1)] = (1/bc.dt + 1/(bc.Re*(bc.dy**2)))	
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i)] = -0.5/(bc.Re*(bc.dy**2))		
				B[(j-1)*(N-2) + (i-1)] = (0.5/(bc.Re*(bc.dx**2)))*uts[i+1][j] + (1/(bc.Re*(bc.dy**2)))*uts[i-1][j]
				B[(j-1)*(N-2) + (i-1)] = B[(j-1)*(N-2) + (i-1)] + (1/bc.dt - 1/(bc.Re*(bc.dy**2)))*uts[i][j]

			elif i == N-2:
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-1)] = (1/bc.dt + 1/(bc.Re*(bc.dy**2)))	
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-2)] = -0.5/(bc.Re*(bc.dy**2))		
				B[(j-1)*(N-2) + (i-1)] = (1/(bc.Re*(bc.dx**2)))*uts[i+1][j] + (0.5/(bc.Re*(bc.dy**2)))*uts[i-1][j]
				B[(j-1)*(N-2) + (i-1)] = B[(j-1)*(N-2) + (i-1)] + (1/bc.dt - 1/(bc.Re*(bc.dy**2)))*uts[i][j]
			
			elif i > 1 and i < N-2: 
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-1)] = (1/bc.dt + 1/(bc.Re*(bc.dy**2)))	
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i)] = -0.5/(bc.Re*(bc.dy**2))
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-2)] = -0.5/(bc.Re*(bc.dy**2))		
				B[(j-1)*(N-2) + (i-1)] = (0.5/(bc.Re*(bc.dy**2)))*uts[i+1][j] + (0.5/(bc.Re*(bc.dy**2)))*uts[i-1][j]
				B[(j-1)*(N-2) + (i-1)] = B[(j-1)*(N-2) + (i-1)] + (1/bc.dt - 1/(bc.Re*(bc.dy**2)))*uts[i][j]				

	tridiag(A,B,X,N,M)

	for j in range(1,M-1):
		for i in range(1,N-1):
			u[i][j] = X[(j-1)*(N-2) + (i-1)]
	u[N-1][:] = 1.0

	#sweep in the X-direction to calculate v_***
	for i in range(1,N-1):	#only internal nodes 
		for j in range(1,M-1):	#only internal nodes
			
			if j == 1:
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-1)] = (1/bc.dt + 1/(bc.Re*(bc.dx**2)))	
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j)] = -0.5/(bc.Re*(bc.dx**2))		
				B[(i-1)*(M-2) + (j-1)] = (0.5/(bc.Re*(bc.dx**2)))*vds[i][j+1] + (1/(bc.Re*(bc.dx**2)))*vds[i][j-1]
				B[(i-1)*(M-2) + (j-1)] = B[(i-1)*(M-2) + (j-1)] + (1/bc.dt - 1/(bc.Re*(bc.dx**2)))*vds[i][j]

			elif j == M-2:
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-1)] = (1/bc.dt + 1/(bc.Re*(bc.dx**2)))	
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-2)] = -0.5/(bc.Re*(bc.dx**2))		
				B[(i-1)*(M-2) + (j-1)] = (1/(bc.Re*(bc.dx**2)))*vds[i][j+1] + (0.5/(bc.Re*(bc.dx**2)))*vds[i][j-1]
				B[(i-1)*(M-2) + (j-1)] = B[(i-1)*(M-2) + (j-1)] + (1/bc.dt - 1/(bc.Re*(bc.dx**2)))*vds[i][j]
			
			elif j > 1 and j < M-2: 
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-1)] = (1/bc.dt + 1/(bc.Re*(bc.dx**2)))	
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j)] = -0.5/(bc.Re*(bc.dx**2))
				A[(i-1)*(M-2) + (j-1)][(i-1)*(M-2) + (j-2)] = -0.5/(bc.Re*(bc.dx**2))		
				B[(i-1)*(M-2) + (j-1)] = (0.5/(bc.Re*(bc.dx**2)))*vds[i][j+1] + (0.5/(bc.Re*(bc.dx**2)))*vds[i][j-1]
				B[(i-1)*(M-2) + (j-1)] = B[(i-1)*(M-2) + (j-1)] + (1/bc.dt - 1/(bc.Re*(bc.dx**2)))*vds[i][j]				

	tridiag(A,B,X,N,M)

	for i in range(1,N-1):
		for j in range(1,M-1):
			vts[i][j] = X[(i-1)*(M-2) + (j-1)]

	#sweep in the Y-direction to calculate v_n+1
	for j in range(1,M-1):	#only internal nodes 
		for i in range(1,N-1):	#only internal nodes
			
			if i == 1:
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-1)] = (1/bc.dt + 1/(bc.Re*(bc.dy**2)))	
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i)] = -0.5/(bc.Re*(bc.dy**2))		
				B[(j-1)*(N-2) + (i-1)] = (0.5/(bc.Re*(bc.dx**2)))*vts[i+1][j] + (1/(bc.Re*(bc.dy**2)))*vts[i-1][j]
				B[(j-1)*(N-2) + (i-1)] = B[(j-1)*(N-2) + (i-1)] + (1/bc.dt - 1/(bc.Re*(bc.dy**2)))*vts[i][j]

			elif i == N-2:
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-1)] = (1/bc.dt + 1/(bc.Re*(bc.dy**2)))	
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-2)] = -0.5/(bc.Re*(bc.dy**2))		
				B[(j-1)*(N-2) + (i-1)] = (1/(bc.Re*(bc.dx**2)))*vts[i+1][j] + (0.5/(bc.Re*(bc.dy**2)))*vts[i-1][j]
				B[(j-1)*(N-2) + (i-1)] = B[(j-1)*(N-2) + (i-1)] + (1/bc.dt - 1/(bc.Re*(bc.dy**2)))*vts[i][j]
			
			elif i > 1 and i < N-2: 
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-1)] = (1/bc.dt + 1/(bc.Re*(bc.dy**2)))	
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i)] = -0.5/(bc.Re*(bc.dy**2))
				A[(j-1)*(N-2) + (i-1)][(j-1)*(N-2) + (i-2)] = -0.5/(bc.Re*(bc.dy**2))		
				B[(j-1)*(N-2) + (i-1)] = (0.5/(bc.Re*(bc.dy**2)))*vts[i+1][j] + (0.5/(bc.Re*(bc.dy**2)))*vts[i-1][j]
				B[(j-1)*(N-2) + (i-1)] = B[(j-1)*(N-2) + (i-1)] + (1/bc.dt - 1/(bc.Re*(bc.dy**2)))*vts[i][j]				

	tridiag(A,B,X,N,M)

	for j in range(1,M-1):
		for i in range(1,N-1):
			v[i][j] = X[(j-1)*(N-2) + (i-1)]

	return 

def tridiag(A,B,X,N,M):
		
	R = int((N-2)*(M-2))

	for i in range(0,R):			

		if i == 0:
			A[i][i+1] = A[i][i+1]/A[i][i]
			B[i] = B[i]/A[i][i]

		elif i < R-1:
			A[i][i+1] = A[i][i+1]/(A[i][i] - A[i][i-1]*A[i-1][i])


		B[i] = (B[i] - A[i][i-1]*B[i-1])/(A[i][i] - A[i][i-1]*A[i-1][i])
		A[i][i-1] = 0
		A[i][i] = 1.0

	for i in range(R-1,-1,-1):
		
		if i == R-1:
			X[i] = B[i]
		
		elif i != R-1: 
			X[i] = B[i] - A[i][i+1]*X[i+1]	

	A = np.zeros([(N-2)*(M-2),(N-2)*(M-2)], dtype = np.float)
	B = np.zeros([(N-2)*(M-2),1], dtype = np.float)
	X = np.zeros([(N-2)*(M-2),1], dtype = np.float)
	return 



