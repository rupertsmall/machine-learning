# Optimise a neural network

import threading
from Queue import Queue
from numpy import *
from create_MEGA_THETA import *
from RLU_forward_backward import *
from get_overlaps import *

# get data from csv file into array
data = genfromtxt('train2.csv', delimiter=',')
 
# get a subset for testing
num_cpus = 4
N = 5*num_cpus	 			# batch size of data subset
y_vals = data[:,0]			# outputs
x_vals = data[:,1:].T			# inputs
#x_vals = (x_vals > 0)*ones(shape(x_vals))	# make inputs 0/1
data_size = size(y_vals)		

# define some NN layers
base_in = 783
xi = array([1525,10])
mangle_upper = 30

# randomly initiate NN values
MT = create_MEGA_THETA(xi)
#MT = genfromtxt('convltn_to_10_backup.csv', delimiter=',')
MTDIM = shape(MT)
DELTA = zeros([MTDIM[0],MTDIM[1]])
 
# learning rate
alpha = 1*N**2
beta = .02
# create some Queues for DELTA, time, good
time_queue = Queue()
good_queue = Queue()
DELTA_queue = Queue()

# initiate queues
time_queue.put(1.0)
good_queue.put(0.0)

success = 0				# ouch
counter = 0				# meh

# execute backprop procedure until accuracy is high
while success < .9999999: # success rate goal

        selection = random.random_integers(0,data_size-1,N)     # stochastic gradient descent

        # get some pixels labels to mangle
        mangle0 = random.randint(0,mangle_upper)
        mangle1 = random.randint(0,mangle_upper)
        random_mangle0 = random.random_integers(0,base_in,mangle0)
        random_mangle1 = random.random_integers(0,base_in,mangle1)

        batch_x_vals = x_vals[:,selection]                      # SGD batch
        batch_y_vals = y_vals[selection]                        # SGD batch

        # mangle pixels, some --> 0 some --> 1  
        batch_x_vals[random_mangle0,:] = 0
        batch_x_vals[random_mangle1,:] = 1

	DELTA_queue.put(DELTA)
	DELTA_queue.task_done() 
	# run backprop optimisation
	threads = []			# use multi-threading
	for i in range(0,N):
		# input vector
		x = batch_x_vals[:,i]
		overlaps = get()	# from get_overlaps import
		x1 = x[overlaps[0]]
		x2 = x[overlaps[1]]
		x4 = x[overlaps[2]]
		x = hstack((x1, x2, x2, x4, x4, x4, x4))
		y = zeros(10)		# output of NN (row-less)
		y[batch_y_vals[i]] = 1
		thr = threading.Thread(target=forwardBackward,
			args=(xi, x, y, MT, time_queue, good_queue, DELTA_queue))
		threads.append(thr)
	
	print 'StArTiNg ThReAdS'
	
	# start all threads
	for i in range(0,N):
		threads[i].start()
	
	# spin until all threads finish
	for i in range(0,N):
		threads[i].join()
        
	good = good_queue.get()
	time = time_queue.get()
	success = good/time
	
	print 'Success: ',success
	
	good_queue.put(good)
	good_queue.task_done()
	time_queue.put(time)
	time_queue.task_done()
  	DELTA = DELTA_queue.get()
	
	print 'Queue size: ',DELTA_queue.qsize()
	
	MT = MT - (alpha/N)*DELTA #- N*beta*MT
	DELTA = 0*DELTA
	if counter == 50:
		savetxt('convltn_to_10.csv', MT, delimiter=',')
		print 'Success: ',success
		print '\n###  SAVED DATA to convltn_to_10.csv  ###\n'
		counter = 0 # reset
	counter += 1

savetxt('convltn_to_10.csv', MT, delimiter=',')
