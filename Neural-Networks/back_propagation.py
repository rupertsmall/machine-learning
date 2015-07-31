# back propagation algorithm

from numpy import *

def back_propagation(y, A, MEGA_THETA, xi):
	# assume y, A, xi are 1-D column vectors (row-less)
	# assume MEGA_THETA is 2-D array
	
	# define useful constants
	L = size(xi)
	a = A[-xi[-1]:][:, newaxis] 
	delta = a - y[:, newaxis]
	DIM = shape(MEGA_THETA)
	DELTA = zeros([DIM[0],DIM[1]])		# matrix of dJ/dTHETA
	rows_A = sum(xi) + L -1
	
	# define index start/end values
	end_index = rows_A - xi[-1]
	start_index = end_index - xi[-2] -1
	D_row_start = DIM[0] - xi[-1]
	D_row_end = DIM[0]
	D_col_start = DIM[1] - xi[-2] -1
	D_col_end = DIM[1]
	
	# execute first step outside for loop (it doesn't involve MEGA_THETA)
	a_prev = a				# save for later
	a = A[start_index:end_index][:,newaxis]	# next layer inwards from end
	DELTA[D_row_start:D_row_end, D_col_start:D_col_end] = kron(delta, a.T) 
	
	# iterate backwards through each later (back propagation)
	for i in range(L-1,1,-1):
		
		# calculate the next delta
		g = a_prev*(1 - a_prev)
		# local matrix for this layer
		local_theta = MEGA_THETA[D_row_start:D_row_end, D_col_start:D_col_end]
		# the derivative in the back-prop algorithm doesn't include the first elmnt
		delta = dot(local_theta.T, g*delta)[1:,:]
		
		
		# now update DELTA indices for next loop
		D_row_end = D_row_start # yes, strange indeed ! 
		D_row_start = D_row_end - xi[i-1]
		D_col_end = D_col_start
		D_col_start = D_col_end - xi[i-2] -1
		
		# update indices to select from A
		end_index = start_index
		start_index = end_index - xi[i-2] -1
		
		# calculate DELTA for next layer
		a_prev = a[1:,:]
		a = A[start_index:end_index]
		DELTA[D_row_start:D_row_end, D_col_start:D_col_end] = kron(delta, a.T)
		
	return DELTA
