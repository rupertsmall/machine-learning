# forward propagation for neural network

from numpy import *
def neural_forward(xi, input, MEGA_THETA=0):
 
	# initialise MEGA_THETA if required, run forward prop, give result
	
    	if size(MEGA_THETA) == 1: 
 	# MEGA_THETA not given, so initialise it
       		MEGA_THETA = create_MEGA_THETA(xi)

    	# define subset of MEGA_THETA
   	row_start = 0
	row_end = xi[1]
    	col_start = 0
    	col_end = xi[0] + 1
    	
    	# subsets of output vectors a0, a1, a2, .. etc
    	L = size(xi)			# layers in NN (including in/out)
    	rows_A = sum(xi) + L -1		
    	A_start_index = 0
    	A_end_index = xi[0]+1
    	A = zeros(rows_A)		# store output from each layer (row-less) 
    	
    	# save input/output vectors into A for backprop
    	a = r_[array([1]), input]
    	A[A_start_index: A_end_index] = a
         	
    	for i in range(2,L):
        	# local_theta is theta for layer i-1
        	local_theta = MEGA_THETA[row_start:row_end,col_start:col_end]
       		a = r_[array([1]), g(dot(local_theta,a))]	# add base unit
       		A_start_index = A_end_index
       		A_end_index = A_start_index + xi[i-1] + 1	# add base unit
       		
        	# for local_theta going into next loop
        	row_start = row_end
        	row_end = row_end + xi[i] + 1
        	col_start = col_end
        	col_end = col_end + xi[i-1] + 1 
        	
        	# save results
        	A[A_start_index:A_end_index] = a
    		
    	# last step due to i=2,3,..,L-1
    	local_theta = MEGA_THETA[row_start:row_end,col_start:col_end]
    	a = g(dot(local_theta,a))
    	A[rows_A - xi[-1]:] = a
    	return A

# # # #

def g(x):
	# sigmoid function for NN nodes
	#g = 1/(1+exp(-x));
	
	# rectified linear units
	g = ones(len(x))*(x>0)
	
	# linearised RLUs
	#g = ones(len(x))*(x>0)
	return g
