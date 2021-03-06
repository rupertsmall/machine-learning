# Optimise a neural network

import threading
from Queue import Queue
from numpy import *
from create_MEGA_THETA import *
from forward_backward import *

# get data from csv file into array
data = genfromtxt('training_data.csv', delimiter=',')
 
# get a subset for testing
num_cpus = 32
N = 100*num_cpus 			# batch size of data subset
y_vals = data[:,0]			# outputs
x_vals = data[:,1:].T			# inputs
x_vals = (x_vals > 0)*x_vals		# make inputs 0/1
data_size = size(y_vals)		

# define some NN layers
xi = array([100,10])
 
# randomly initiate NN values
MT = create_MEGA_THETA(xi)
MTDIM = shape(MT)
DELTA = zeros([MTDIM[0],MTDIM[1]])
 
# learning rate
alpha = 100*N
 
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
while success < .99: # aim for success rate greater than .999999
	selection = random.random_integers(0,data_size-1,N)	# stochastic gradient descent
	batch_x_vals = x_vals[:,selection]			# SGD batch
	batch_y_vals = y_vals[selection]			# SGD batch
	DELTA_queue.put(DELTA)
	DELTA_queue.task_done() 
	# run backprop optimisation
	threads = []			# use multi-threading
	for i in range(0,N):
		# input vector
		x = x_vals[:,i]
		y = zeros(10)		# output of NN (row-less)
		y[y_vals[i]] = 1
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
		DELTA_queue.join()
        
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
	MT = MT - (alpha/N)*DELTA
	DELTA = 0*DELTA
	if counter == 1000:
		savetxt('MT_ALLto10.csv', MT, delimiter=',')
		print '\n###  SAVED DATA to MT_ALLto10.csv  ###\n'
		counter = 0 # reset
	counter += 1

savetxt('MT_ALLto10.csv', MT, delimiter=',')
