#
# neural network optimisation classes for facial keypoints recognition
# rupert small, august 2015
#

from Queue import Queue
from create_MEGA_THETA import *
from forward_backward import *
from neural_forward import *
from back_propagation import *
import threading

class fk_zone1(object):
	'''create instance of neural network for zone 1 of facial keypoints optimiser'''

	def __init__(self, inlayer, outlayer, alpha = 1, beta = 0.4, mangle_upper = 0, save = 10, threads = 40):
		'''initiate queues and primary variables'''
		self.zone2_queue = Queue()
		self.DELTA_queue = Queue()
		self.errors_queue = Queue()

		# initiate NN variables
		self.counter = 0
		self.mangle_upper = mangle_upper
		self.file_out = 'MT_zoneA.csv'
		self.file_name = 'MT_zoneA_backup.csv'
		#self.MT = genfromtxt(self.file_name, delimiter=',')
		self.xi = [inlayer, outlayer]
		self.MT = create_MEGA_THETA(self.xi)
		self.MTDIM = shape(self.MT)
		self.DELTA = zeros([self.MTDIM[0],self.MTDIM[1]])
		self.alpha = alpha
		self.beta = beta
		self.save_when = save
		self.num_threads = threads
		self.threads = []

	def optimise_network(self, all_x, all_y):
		'''run backpropagation for digit = self.digit and save results'''
		
		self.zone2init = fk_zone2(self.zone2_queue)
		self.z2thread = threading.Thread(target=self.zone2init.start)
		#self.z2thread.start()
		self.data_size = shape(all_x)[1]
		# execute backprop procedure forever
		while True:

			# get some pixels' labels to mangle
			self.selection = random.random_integers(0,self.data_size-1, self.num_threads)
			#self.mangle0 = random.randint(0,self.mangle_upper)
			#self.mangle1 = random.randint(0,self.mangle_upper)
			#self.random_mangle0 = random.random_integers(0,self.xi[0]-1, self.mangle0)
			#self.random_mangle1 = random.random_integers(0,self.xi[0]-1, self.mangle1)
			# mangle pixels, some --> 0 some --> 255	
			#self.x[self.random_mangle0] = 0
			#self.x[self.random_mangle1] = 255

			self.x = all_x[:,self.selection]
			self.y = all_y[:,self.selection]

			# spin up some threads
			for i in range(0,self.num_threads):
				th = threading.Thread(target=forwardBackward, args=(self.xi, self.x[:,i], self.y[:,i], self.MT, self.DELTA_queue, self.errors_queue, self.zone2_queue))
				self.threads.append(th)

			print 'spinning'

			# pause until finished			
			for i in range(0, self.num_threads): self.threads[i].start()
			for i in range(0, self.num_threads): self.threads[i].join()
			
			print 'joining'

			# debugging:	
			# print self.y		
			# print self.zone1out	
			print amax(self.MT)
			# print amin(self.MT)
			# print amax(self.x)
			# print amin(self.y)
			# print self.A[-30:]
			
			self.errors = self.errors_queue.get()/self.num_threads
			# run back propagation
		  	self.DELTA = back_propagation(self.y, self.A, self.MT, self.xi)
			self.MT = self.MT - self.alpha*self.DELTA/self.num_threads - self.beta*self.MT
			
			if self.counter == self.save_when:
				savetxt(self.file_out, self.MT, delimiter=',')
				print 'saved zone1 data to ' + self.file_out + '\t',
				print 'errors zone1: ' + str(sum(self.errors)) + '\tmax error arg: ' + str(argmax(self.errors)) + '\t MEAE Z1: ' + str(int(amax(self.errors))) + '\tmin error arg: ' + str(argmin(self.errors)) + '\t MinEAE Z1: ' + str(int(amin(self.errors)))

				self.counter = 0 # reset
			self.counter += 1

			self.DELTA = 0*self.DELTA
			self.errors = 0*self.errors

		#savetxt(self.file_out, self.MT, delimiter=',')


class fk_zone2(object):
	'''create instance of neural network for zone 2 of facial keypoints optimiser'''
	
	def __init__(self, zone2queue, alpha = .1, beta = .01, save = 10000):
				
		self.counter = 0
		self.alpha = alpha
		self.beta = beta
                self.file_out = 'MT_zoneB.csv'
                self.file_name = 'MT_zoneB_backup.csv'
                #self.MT = genfromtxt(self.file_name, delimiter=',')
                self.xi = [30, 30]
                self.MT = create_MEGA_THETA(self.xi)
                self.MTDIM = shape(self.MT)
              	self.DELTA = zeros([self.MTDIM[0],self.MTDIM[1]])
		self.zone2queue = zone2queue
		self.save_when = save

	def start(self):
		'''start learning process for zone2 of facial keypoints analysis'''
		
		while False:
			
			self.data = self.zone2queue.get()
			self.zone2queue.task_done()
			self.x = self.data[:,0]
			self.y = self.data[:,1]
			self.A = neural_forward(self.xi, self.x, self.MT)
			self.DELTA = back_propagation(self.y, self.A, self.MT, self.xi)
			self.errors = sqrt((self.A[-self.xi[1]:]-self.y)**2)
			self.MT = self.MT - self.alpha*self.DELTA - self.beta*self.MT
                        self.DELTA = 0*self.DELTA

                        if self.counter == self.save_when:
                                savetxt(self.file_out, self.MT, delimiter=',')
                                print 'saved zone2 data to ' + self.file_out + '\t',
                                print 'errors zone2: ' + str(sum(self.errors)) + '\tmax error arg: ' + str(argmax(self.errors)) + '\t MEAE Z2: ' + str(int(amax(self.errors))) + '\tmin error arg: ' + str(argmin(self.errors)) + '\t MinEAE Z2: ' + str(int(amin(self.errors)))

                                self.counter = 0 # reset
                        self.counter += 1
