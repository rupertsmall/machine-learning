# Optimise a neural network

import threading
from Queue import Queue
from numpy import *
from neural_forward_multithreaded import *

# get data from csv file into array
data = genfromtxt('test2.csv', delimiter=',')
 
# get a subset for testing
x_batch = data.T			# inputs
x_batch = (x_batch > 0)*x_batch		# make inputs 0/1
N = shape(x_batch)[1]
# define NN layers
xi = array([784**2,2])

# import trained neural networks 
MT0 = genfromtxt('MT_ZERO_backup.csv', delimiter=',')
MT1 = genfromtxt('MT_ONE_backup.csv', delimiter=',')
MT2 = genfromtxt('MT_TWO_backup.csv', delimiter=',')
MT3 = genfromtxt('MT_THREE_backup.csv', delimiter=',')
MT4 = genfromtxt('MT_FOUR_backup.csv', delimiter=',')
MT5 = genfromtxt('MT_FIVE_backup.csv', delimiter=',')
MT6 = genfromtxt('MT_SIX_backup.csv', delimiter=',')
MT7 = genfromtxt('MT_SEVEN_backup.csv', delimiter=',')
MT8 = genfromtxt('MT_EIGHT_backup.csv', delimiter=',')
MT9 = genfromtxt('MT_NINE_backup.csv', delimiter=',')
 
for k in range(N-1, N):
	A_queue = Queue()       # queue for output values
	zero2nine = zeros(10)	# vector for true output layer
	A_queue.put(zero2nine)	# pass in vector
	threads = []        	# store threads
	for i in range(0,10):
		# input vector
		x = x_batch[:,k]
		x = kron(x,x)
		# find a better way to reference MT arrays...
		if i==0:
			thr = threading.Thread(target=neural_forward_multithreaded,
				args=(xi, x, A_queue, MT0, i))
			threads.append(thr)
		if i==1:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT1, i))
                        threads.append(thr)
                
		if i==2:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT2, i))
                        threads.append(thr)
                
		if i==3:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT3, i))
                        threads.append(thr)
                
		if i==4:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT4, i))
                        threads.append(thr)
                
		if i==5:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT5, i))
                        threads.append(thr)
                
		if i==6:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT6, i))
                        threads.append(thr)
                
		if i==7:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT7, i))
                        threads.append(thr)
                
		if i==8:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT8, i))
                        threads.append(thr)
                
		if i==9:
                        thr = threading.Thread(target=neural_forward_multithreaded,
                                args=(xi, x, A_queue, MT9, i))
                        threads.append(thr)
                

	# start all threads
	for i in range(0,10):
		threads[i].start()
		#print 'StArTiNg ThReAd: ',i
	
	# spin until all threads finish
	for i in range(0,10):
		threads[i].join()
       
	zero2nine = A_queue.get()
	print str(k+1)+','+str(argmax(zero2nine))
