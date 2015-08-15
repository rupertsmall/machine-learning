#
# neural network optimisation classes
# rupert small, august 2015
#

from Queue import Queue
from numpy import *
from create_MEGA_THETA import *
from RLU_forward_backward import *
import threading


class dr_singles(object):
	'''create instance of dr_singles neural network'''

	def __init__(self, alpha, beta, base_in, mangle_upper, num_cpus, digit):
		'''initiate queues and primary variables'''

		# create some Queues for DELTA, time, good
		self.time_queue = Queue()
		self.good_queue = Queue()
		self.DELTA_queue = Queue()
		# initiate queues
		self.time_queue.put(1.0)
		self.good_queue.put(0.0)
	
		# initiate NN variables	
		self.success = 0
		self.counter = 0
		self.base_in = base_in
		self.mangle_upper = mangle_upper
		self.digits = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE']
		#self.file_name = 'MT_' + self.digits[digit] + '_backup.csv'
		self.file_out = 'MT_' + self.digits[digit] + '.csv'
		#self.MT = genfromtxt(self.file_name, delimiter=',')
		#self.MTDIM = shape(self.MT)
		#self.DELTA = zeros([self.MTDIM[0],self.MTDIM[1]])
		self.N = 2*num_cpus
		self.alpha = alpha
		self.beta = beta
		self.digit = digit


	def optimise_network(self, xi, all_x, all_y):
		'''run backpropagation for digit = self.digit and save results'''

		self.MT = create_MEGA_THETA(xi)
		self.MTDIM = shape(self.MT)
		self.DELTA = zeros([self.MTDIM[0],self.MTDIM[1]])
		
		self.data_size = size(all_y)
		# execute backprop procedure until accuracy is high
		while self.success < .99999: # success rate goal
			self.selection = random.random_integers(0,self.data_size-1,self.N) # stochastic gradient descent
				
			# get some pixels' labels to mangle
			self.mangle0 = random.randint(0,self.mangle_upper)
			self.mangle1 = random.randint(0,self.mangle_upper)
			self.random_mangle0 = random.random_integers(0,self.base_in, self.mangle0)
			self.random_mangle1 = random.random_integers(0,self.base_in, self.mangle1)
			
			self.batch_x_vals = all_x[:,self.selection]	# SGD batch
			self.batch_y_vals = all_y[self.selection]	# SGD batch
			
			# mangle pixels, some --> 0 some --> 1	
			self.batch_x_vals[self.random_mangle0,:] = 0
			self.batch_x_vals[self.random_mangle1,:] = 1
			
			self.DELTA_queue.put(self.DELTA)
			self.DELTA_queue.task_done() 
			# run backprop optimisation w/ multi-threading
			self.threads = []
			
			# set NN input/output layers
			for i in range(0,self.N):
				# input vector
				self.x = self.batch_x_vals[:,i]
				self.x = kron(self.x, self.x)
				self.y = zeros(2)		# output of NN (row-less)
				if self.batch_y_vals[i] == self.digit:
					self.y[0] = 1	# indicate digit
				else:
					self.y[1] = 1	# indicate NOT digit
				self.thr = threading.Thread(target=forwardBackward,
					args=(xi, self.x, self.y, self.MT, self.time_queue, self.good_queue, self.DELTA_queue))
				self.threads.append(self.thr)

			# start all threads
			for i in range(0,self.N):
				self.threads[i].start()
			
			# spin until all threads finish
			for i in range(0,self.N):
				self.threads[i].join()
        			self.DELTA_queue.join()

			self.good = self.good_queue.get()
			self.time = self.time_queue.get()
			self.success = self.good/self.time

			print 'Success['+ str(self.digit) +']: ', self.success

			self.good_queue.put(self.good)
			self.good_queue.task_done()
			self.time_queue.put(self.time)
			self.time_queue.task_done()
		  	self.DELTA = self.DELTA_queue.get()
			self.MT = self.MT - (self.alpha/self.N)*self.DELTA - self.beta*self.MT
			self.DELTA = 0*self.DELTA
			if self.counter == 50:
				savetxt(self.file_out, self.MT, delimiter=',')
				print 'Success ['+ str(self.digit) +']: ', self.success
				print '###  SAVED DATA to '+ self.file_out +'  ###'
				self.counter = 0 # reset
			self.counter += 1

		savetxt(self.file_out, self.MT, delimiter=',')
