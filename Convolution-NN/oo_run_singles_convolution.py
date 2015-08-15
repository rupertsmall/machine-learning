#
# optimise a neural network for image recognition
# rupert small, august 2015
#

from numpy import genfromtxt
from oo_dr_singles_convolution import *
from get_overlaps import *
import threading

# initiate data
data = genfromtxt('train2.csv', delimiter=',')
num_cpus = 30		# multi-threading
y_vals = data[:,0]	# outputs
x_vals = data[:,1:]	# inputs
alpha = 20*num_cpus**2	# learning
beta = .02		# regularisation
base_in = 783		# number of independent points
input_layer = 1525 	# number of input nuerons
mangle_upper = 35	# replacement/mangling
xi = array([input_layer,2])
dim = shape(x_vals)
x = zeros([dim[0], 1525])
threads= []

# convolution
overlaps = get()
for i in range(0,dim[0]):
	temp = x_vals[0,:]
	x1 = temp[overlaps[0]]
	x2 = temp[overlaps[1]]
	x4 = temp[overlaps[2]]		
	x[i,:] = hstack((x1, x2, x2, x4, x4, x4, x4))
	x_vals = delete(x_vals, 0, 0)

x = x.T
	
# initiate classes
zero = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 0)
one = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 1)
two  = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 2)
three = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 3)
four = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 4)
five = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 5)
six  = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 6)
seven = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 7)
eight = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 8)
nine = dr_singles(alpha, beta, base_in, mangle_upper, num_cpus, 9)


# run
thr = threading.Thread(target=zero.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=one.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=two.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=three.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=four.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=five.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=six.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=seven.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=eight.optimise_network, args=(xi, x, y_vals))
threads.append(thr)
thr = threading.Thread(target=nine.optimise_network, args=(xi, x, y_vals))
threads.append(thr)

for i in range(0,len(threads)):
	threads[i].start()
